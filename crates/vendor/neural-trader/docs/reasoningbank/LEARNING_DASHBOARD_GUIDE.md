# ReasoningBank Learning Dashboard Guide

## Overview

The ReasoningBank Learning Dashboard provides comprehensive real-time visualization and analytics for E2B swarm learning processes. It tracks learning curves, decision quality, pattern discovery, agent skills, and provides predictive analytics and optimization recommendations.

## Features

### 📊 Real-Time Metrics
- **Learning Curve**: Accuracy progression over episodes
- **Decision Quality**: Quality score trends over time
- **Pattern Growth**: Pattern discovery progression
- **Agent Skills**: Skill matrices for each agent
- **Knowledge Graph**: Pattern relationships and connections

### 📈 Comparative Analysis
- **Topology Comparison**: Learning performance by topology type
- **Strategy Comparison**: Effectiveness of different learning strategies
- **Agent Comparison**: Performance metrics across agents

### 🔮 Predictive Analytics
- **Convergence Prediction**: Estimated episodes to target accuracy
- **Bottleneck Analysis**: Identify learning bottlenecks
- **Optimization Recommendations**: Actionable improvement suggestions

### 💾 Export Formats
- **HTML Dashboard**: Interactive charts with Chart.js
- **Markdown Reports**: Documentation-ready reports
- **JSON Data**: Raw data for further analysis
- **ASCII Charts**: Terminal-friendly visualizations

## Installation

The dashboard is included in the neural-trader package. No additional installation required.

```bash
# Ensure you have the CLI installed
npm install -g neural-trader

# Or run from project directory
node scripts/e2b-swarm-cli.js learning --help
```

## CLI Commands

### Live Dashboard

Display real-time learning dashboard in terminal:

```bash
# Live updates with default settings (2s interval, 60s duration)
node scripts/e2b-swarm-cli.js learning dashboard --live

# Custom interval and duration
node scripts/e2b-swarm-cli.js learning dashboard --live -i 5000 -d 120000

# Load from custom data file
node scripts/e2b-swarm-cli.js learning dashboard --live -s ./my-data.json
```

### Generate HTML Dashboard

Create interactive HTML dashboard:

```bash
# Generate and open in browser
node scripts/e2b-swarm-cli.js learning dashboard

# Save to specific location
node scripts/e2b-swarm-cli.js learning dashboard -o ./reports/dashboard.html

# Use custom data source
node scripts/e2b-swarm-cli.js learning dashboard -s ./my-data.json
```

### Generate Reports

Create analytics reports in various formats:

```bash
# HTML report
node scripts/e2b-swarm-cli.js learning report --format html

# Markdown report
node scripts/e2b-swarm-cli.js learning report --format markdown

# JSON data export
node scripts/e2b-swarm-cli.js learning report --format json

# Custom output path
node scripts/e2b-swarm-cli.js learning report --format html -o ./my-report.html
```

### View Statistics

Quick statistics display:

```bash
# Show overall stats
node scripts/e2b-swarm-cli.js learning stats

# Show specific agent stats
node scripts/e2b-swarm-cli.js learning stats --agent agent-001

# Load from file
node scripts/e2b-swarm-cli.js learning stats -s ./my-data.json
```

### Analytics

Display learning analytics and recommendations:

```bash
# Show full analytics
node scripts/e2b-swarm-cli.js learning analytics

# Use custom data
node scripts/e2b-swarm-cli.js learning analytics -s ./my-data.json
```

### Export Data

Export learning data for analysis:

```bash
# Export as JSON
node scripts/e2b-swarm-cli.js learning export --format json

# Custom output path
node scripts/e2b-swarm-cli.js learning export -o ./data/learning-export.json

# Export from custom source
node scripts/e2b-swarm-cli.js learning export -s ./source-data.json -o ./export.json
```

## Programmatic Usage

### Basic Example

```javascript
const { LearningDashboard } = require('./src/reasoningbank/learning-dashboard');

// Create dashboard instance
const dashboard = new LearningDashboard({
  updateInterval: 1000,
  historySize: 100,
  targetAccuracy: 0.95
});

// Update with learning data
dashboard.updateMetrics({
  learningCurve: [
    { episode: 1, accuracy: 0.45 },
    { episode: 2, accuracy: 0.52 },
    // ... more episodes
  ],
  decisionQuality: [
    { score: 0.6, timestamp: '2025-01-01T00:00:00Z' },
    // ... more data points
  ],
  patternGrowth: [
    { count: 5, timestamp: '2025-01-01T00:00:00Z' },
    // ... more data points
  ],
  agentSkills: {
    'agent-001': {
      'pattern-recognition': 0.85,
      'decision-making': 0.78,
      'exploration': 0.65
    }
  }
});

// Display visualizations
console.log(await dashboard.displayLearningCurve());
console.log(await dashboard.displayDecisionQuality());
console.log(await dashboard.predictConvergence());

// Export
await dashboard.exportHTML('./dashboard.html');
await dashboard.exportMarkdown('./report.md');
await dashboard.exportJSON('./data.json');
```

### Advanced Example with Custom Data

```javascript
const { LearningDashboard } = require('./src/reasoningbank/learning-dashboard');
const fs = require('fs').promises;

async function analyzeLearning(dataPath) {
  // Load learning data
  const rawData = await fs.readFile(dataPath, 'utf8');
  const data = JSON.parse(rawData);

  // Create dashboard
  const dashboard = new LearningDashboard({
    targetAccuracy: 0.90
  });

  dashboard.updateMetrics(data);

  // Generate comprehensive report
  const report = await dashboard.generateCompleteReport();
  console.log(report);

  // Get specific analytics
  const convergence = await dashboard.predictConvergence();
  const bottlenecks = await dashboard.identifyBottlenecks();
  const recommendations = await dashboard.recommendOptimizations();

  // Export for documentation
  await dashboard.exportHTML('./docs/dashboard.html');
  await dashboard.exportMarkdown('./docs/analysis.md');

  return {
    convergence,
    bottlenecks,
    recommendations
  };
}

analyzeLearning('./learning-data.json');
```

## Data Format

### Input Data Structure

```json
{
  "learningCurve": [
    {
      "episode": 1,
      "accuracy": 0.45,
      "loss": 0.55,
      "timestamp": "2025-01-01T00:00:00Z"
    }
  ],
  "decisionQuality": [
    {
      "score": 0.6,
      "confidence": 0.7,
      "timestamp": "2025-01-01T00:00:00Z"
    }
  ],
  "patternGrowth": [
    {
      "count": 5,
      "newPatterns": 2,
      "timestamp": "2025-01-01T00:00:00Z"
    }
  ],
  "agentSkills": {
    "agent-001": {
      "pattern-recognition": 0.85,
      "decision-making": 0.78,
      "exploration": 0.65,
      "exploitation": 0.92,
      "coordination": 0.70
    }
  },
  "knowledgeGraph": {
    "nodes": [
      {
        "id": "p1",
        "pattern": "High volatility → Risk reduction",
        "confidence": 0.92
      }
    ],
    "edges": [
      {
        "from": "p1",
        "to": "p2",
        "weight": 0.65
      }
    ]
  },
  "topologies": [
    {
      "topology": "Mesh",
      "accuracy": 0.89,
      "episodes": 45
    }
  ],
  "strategies": [
    {
      "strategy": "Q-Learning",
      "effectiveness": 0.84,
      "convergence": 42
    }
  ],
  "agents": [
    {
      "agent": "Agent-001",
      "performance": 0.87,
      "patterns": 15
    }
  ]
}
```

## Generating Demo Data

Create demo data for testing:

```bash
# Generate demo data
node src/reasoningbank/demo-data-generator.js

# Generate to custom location
node src/reasoningbank/demo-data-generator.js ./my-demo-data.json

# Use in dashboard
node scripts/e2b-swarm-cli.js learning dashboard -s ./my-demo-data.json
```

## ASCII Chart Examples

### Learning Curve
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

### Agent Skills Bar Chart
```
Agent Skills Matrix (agent-001)
================================

pattern-recognition │█████████████████████████████████████████████ 0.85
decision-making     │██████████████████████████████████████ 0.78
exploration         │████████████████████████████ 0.65
exploitation        │████████████████████████████████████████████████ 0.92
coordination        │███████████████████████████████ 0.70
```

## HTML Dashboard Features

The generated HTML dashboard includes:

- 📊 **Interactive Charts**: Powered by Chart.js
- 🎨 **Beautiful Design**: Gradient backgrounds and modern UI
- 📱 **Responsive Layout**: Works on all screen sizes
- 📈 **Multiple Chart Types**: Line, bar, radar, and more
- 🔄 **Real-time Data**: Auto-updates with new metrics
- 📊 **Statistics Cards**: Key metrics at a glance
- 💡 **Analytics Section**: Predictions and recommendations

## Tips & Best Practices

### 1. Data Collection
- Collect metrics at regular intervals (every episode or decision)
- Include timestamps for accurate time-series analysis
- Track both success and failure patterns

### 2. Visualization Selection
- Use line charts for trends over time
- Use bar charts for comparisons
- Use heatmaps for multi-dimensional data
- Use scatter plots for correlation analysis

### 3. Performance Optimization
- Limit history size for large datasets (use `historySize` option)
- Export to HTML for interactive analysis of large datasets
- Use JSON export for data processing pipelines

### 4. Interpretation
- Look for plateaus in learning curves (may indicate need for parameter tuning)
- Monitor pattern growth rate (slowing = need for exploration)
- Check agent skill variance (low = need for specialization)
- Review convergence predictions regularly

## Troubleshooting

### No Data Available
```bash
# Generate demo data first
node src/reasoningbank/demo-data-generator.js ./demo-data.json

# Then use it
node scripts/e2b-swarm-cli.js learning dashboard -s ./demo-data.json
```

### Charts Not Displaying
- Ensure Chart.js CDN is accessible
- Check browser console for errors
- Verify data format matches expected structure

### Slow Performance
- Reduce `historySize` in dashboard options
- Increase `updateInterval` for live mode
- Use JSON export for large dataset analysis

## Examples

### Complete Workflow

```bash
# 1. Generate demo data
node src/reasoningbank/demo-data-generator.js ./learning-data.json

# 2. View quick stats
node scripts/e2b-swarm-cli.js learning stats -s ./learning-data.json

# 3. Generate HTML dashboard
node scripts/e2b-swarm-cli.js learning dashboard -s ./learning-data.json

# 4. Create markdown report
node scripts/e2b-swarm-cli.js learning report --format markdown -s ./learning-data.json

# 5. View analytics
node scripts/e2b-swarm-cli.js learning analytics -s ./learning-data.json

# 6. Export for further analysis
node scripts/e2b-swarm-cli.js learning export -s ./learning-data.json -o ./export.json
```

## API Reference

### LearningDashboard Class

#### Constructor Options
```javascript
{
  updateInterval: 1000,    // Update interval in ms
  historySize: 100,        // Max history points to keep
  targetAccuracy: 0.95     // Target accuracy for predictions
}
```

#### Methods

- `displayLearningCurve(data?)` - Display learning curve chart
- `displayDecisionQuality(data?)` - Display decision quality chart
- `displayPatternGrowth(data?)` - Display pattern growth chart
- `displayAgentSkills(agentId?)` - Display agent skills
- `displayKnowledgeGraph()` - Display knowledge graph
- `compareTopologies(data?)` - Compare topologies
- `compareStrategies(data?)` - Compare strategies
- `compareAgents(data?)` - Compare agents
- `predictConvergence(data?)` - Predict convergence
- `identifyBottlenecks(data?)` - Identify bottlenecks
- `recommendOptimizations()` - Get recommendations
- `updateMetrics(data)` - Update metrics with new data
- `exportHTML(path)` - Export as HTML
- `exportMarkdown(path)` - Export as Markdown
- `exportJSON(path)` - Export as JSON
- `generateCompleteReport()` - Generate full report

## Support

For issues or questions:
- GitHub Issues: [neural-trader/issues](https://github.com/ruvnet/neural-trader/issues)
- Documentation: `/docs/reasoningbank/`

## License

MIT License - see LICENSE file for details
