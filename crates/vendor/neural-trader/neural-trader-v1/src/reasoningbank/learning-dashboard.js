/**
 * ReasoningBank E2B Swarm Learning Dashboard
 * Real-time learning metrics visualization and analytics
 */

const fs = require('fs').promises;
const path = require('path');

/**
 * ASCII Chart Generator for terminal display
 */
class ASCIIChart {
  /**
   * Generate line chart in ASCII
   */
  static lineChart(data, options = {}) {
    const {
      width = 60,
      height = 15,
      title = '',
      xlabel = '',
      ylabel = '',
      showLegend = true
    } = options;

    if (!data || data.length === 0) return 'No data available';

    const values = data.map(d => d.value || d);
    const labels = data.map(d => d.label || '');

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    let chart = '';
    if (title) chart += `\n${title}\n${'='.repeat(title.length)}\n\n`;

    // Y-axis and plot area
    for (let y = height; y >= 0; y--) {
      const value = min + (y / height) * range;
      const label = value.toFixed(2).padStart(8);

      let line = `${label} │`;

      for (let x = 0; x < values.length; x++) {
        const normalized = (values[x] - min) / range;
        const plotY = Math.round(normalized * height);

        if (plotY === y) {
          line += '●';
        } else if (x > 0) {
          const prevNormalized = (values[x - 1] - min) / range;
          const prevY = Math.round(prevNormalized * height);

          if ((plotY < y && prevY >= y) || (plotY >= y && prevY < y)) {
            line += '│';
          } else {
            line += ' ';
          }
        } else {
          line += ' ';
        }
      }

      chart += line + '\n';
    }

    // X-axis
    chart += '         └' + '─'.repeat(values.length) + '\n';

    if (xlabel) {
      chart += `         ${xlabel}\n`;
    }

    if (ylabel) {
      chart = `${ylabel}\n${chart}`;
    }

    return chart;
  }

  /**
   * Generate bar chart in ASCII
   */
  static barChart(data, options = {}) {
    const {
      width = 50,
      title = '',
      showValues = true
    } = options;

    if (!data || data.length === 0) return 'No data available';

    const items = data.map(d => ({
      label: d.label || d.name || '',
      value: d.value || 0
    }));

    const maxValue = Math.max(...items.map(i => i.value));
    const maxLabelLen = Math.max(...items.map(i => i.label.length));

    let chart = '';
    if (title) chart += `\n${title}\n${'='.repeat(title.length)}\n\n`;

    items.forEach(item => {
      const barLen = Math.round((item.value / maxValue) * width);
      const label = item.label.padEnd(maxLabelLen);
      const bar = '█'.repeat(barLen);
      const value = showValues ? ` ${item.value.toFixed(2)}` : '';

      chart += `${label} │${bar}${value}\n`;
    });

    return chart;
  }

  /**
   * Generate heatmap in ASCII
   */
  static heatmap(data, options = {}) {
    const {
      title = '',
      rowLabels = [],
      colLabels = [],
      colorScale = [' ', '░', '▒', '▓', '█']
    } = options;

    if (!data || data.length === 0) return 'No data available';

    const flatValues = data.flat();
    const min = Math.min(...flatValues);
    const max = Math.max(...flatValues);
    const range = max - min || 1;

    let chart = '';
    if (title) chart += `\n${title}\n${'='.repeat(title.length)}\n\n`;

    // Column headers
    if (colLabels.length > 0) {
      const maxRowLabel = Math.max(...rowLabels.map(l => l.length), 0);
      chart += ' '.repeat(maxRowLabel + 2);
      chart += colLabels.map(l => l.substring(0, 3).padEnd(3)).join(' ') + '\n';
    }

    // Data rows
    data.forEach((row, i) => {
      const rowLabel = rowLabels[i] || '';
      chart += rowLabel.padEnd(10) + ' │';

      row.forEach(value => {
        const normalized = (value - min) / range;
        const colorIndex = Math.floor(normalized * (colorScale.length - 1));
        chart += colorScale[colorIndex].repeat(3) + ' ';
      });

      chart += '\n';
    });

    return chart;
  }

  /**
   * Generate scatter plot in ASCII
   */
  static scatterPlot(data, options = {}) {
    const {
      width = 60,
      height = 20,
      title = '',
      xlabel = '',
      ylabel = ''
    } = options;

    if (!data || data.length === 0) return 'No data available';

    const xValues = data.map(d => d.x);
    const yValues = data.map(d => d.y);

    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    // Create plot grid
    const grid = Array(height).fill(0).map(() => Array(width).fill(' '));

    // Plot points
    data.forEach(point => {
      const x = Math.round(((point.x - xMin) / xRange) * (width - 1));
      const y = Math.round(((point.y - yMin) / yRange) * (height - 1));

      if (x >= 0 && x < width && y >= 0 && y < height) {
        grid[height - 1 - y][x] = '●';
      }
    });

    let chart = '';
    if (title) chart += `\n${title}\n${'='.repeat(title.length)}\n\n`;

    // Render grid with axes
    grid.forEach((row, i) => {
      const yValue = yMin + ((height - 1 - i) / (height - 1)) * yRange;
      const yLabel = yValue.toFixed(1).padStart(6);
      chart += `${yLabel} │${row.join('')}\n`;
    });

    chart += '       └' + '─'.repeat(width) + '\n';
    if (xlabel) chart += `        ${xlabel}\n`;

    return chart;
  }
}

/**
 * Main Learning Dashboard
 */
class LearningDashboard {
  constructor(options = {}) {
    this.options = {
      updateInterval: 1000,
      historySize: 100,
      ...options
    };

    this.metrics = {
      learningCurve: [],
      decisionQuality: [],
      patternGrowth: [],
      agentSkills: new Map(),
      knowledgeGraph: { nodes: [], edges: [] }
    };

    this.comparisons = {
      topologies: [],
      strategies: [],
      agents: []
    };
  }

  /**
   * Real-time Metrics Display
   */

  async displayLearningCurve(data = null) {
    const curveData = data || this.metrics.learningCurve;

    if (curveData.length === 0) {
      return 'No learning data available yet';
    }

    const chartData = curveData.map((point, i) => ({
      label: point.episode || i,
      value: point.accuracy || point.value || 0
    }));

    return ASCIIChart.lineChart(chartData, {
      title: 'Learning Curve - Accuracy Over Episodes',
      xlabel: 'Episodes',
      ylabel: 'Accuracy',
      height: 15
    });
  }

  async displayDecisionQuality(data = null) {
    const qualityData = data || this.metrics.decisionQuality;

    if (qualityData.length === 0) {
      return 'No decision quality data available';
    }

    const chartData = qualityData.map((point, i) => ({
      label: i,
      value: point.score || point.value || 0
    }));

    return ASCIIChart.lineChart(chartData, {
      title: 'Decision Quality Score Trends',
      xlabel: 'Time',
      ylabel: 'Quality Score',
      height: 12
    });
  }

  async displayPatternGrowth(data = null) {
    const patternData = data || this.metrics.patternGrowth;

    if (patternData.length === 0) {
      return 'No pattern growth data available';
    }

    const chartData = patternData.map((point, i) => ({
      label: i,
      value: point.count || point.patterns || point.value || 0
    }));

    return ASCIIChart.lineChart(chartData, {
      title: 'Pattern Discovery Over Time',
      xlabel: 'Episodes',
      ylabel: 'Patterns Discovered',
      height: 12
    });
  }

  async displayAgentSkills(agentId = null) {
    const skills = agentId
      ? this.metrics.agentSkills.get(agentId)
      : Array.from(this.metrics.agentSkills.entries())[0]?.[1];

    if (!skills) {
      return 'No agent skill data available';
    }

    const skillData = Object.entries(skills).map(([skill, value]) => ({
      label: skill,
      value: value
    }));

    return ASCIIChart.barChart(skillData, {
      title: `Agent Skills Matrix ${agentId ? `(${agentId})` : ''}`,
      showValues: true
    });
  }

  async displayKnowledgeGraph() {
    const { nodes, edges } = this.metrics.knowledgeGraph;

    if (nodes.length === 0) {
      return 'No knowledge graph data available';
    }

    let output = '\nKnowledge Graph - Pattern Relationships\n';
    output += '==========================================\n\n';

    // Display nodes
    output += 'Patterns:\n';
    nodes.forEach((node, i) => {
      output += `  ${i + 1}. ${node.pattern || node.name || node.id}\n`;
      if (node.confidence) {
        output += `     Confidence: ${(node.confidence * 100).toFixed(1)}%\n`;
      }
    });

    // Display edges
    if (edges.length > 0) {
      output += '\nRelationships:\n';
      edges.forEach(edge => {
        const from = nodes.find(n => n.id === edge.from)?.pattern || edge.from;
        const to = nodes.find(n => n.id === edge.to)?.pattern || edge.to;
        const weight = edge.weight ? ` (${edge.weight.toFixed(2)})` : '';
        output += `  ${from} → ${to}${weight}\n`;
      });
    }

    return output;
  }

  /**
   * Comparative Views
   */

  async compareTopologies(topologyData = null) {
    const data = topologyData || this.comparisons.topologies;

    if (data.length === 0) {
      return 'No topology comparison data available';
    }

    const chartData = data.map(t => ({
      label: t.topology || t.name,
      value: t.accuracy || t.performance || 0
    }));

    return ASCIIChart.barChart(chartData, {
      title: 'Learning Performance by Topology',
      showValues: true
    });
  }

  async compareStrategies(strategyData = null) {
    const data = strategyData || this.comparisons.strategies;

    if (data.length === 0) {
      return 'No strategy comparison data available';
    }

    const chartData = data.map(s => ({
      label: s.strategy || s.name,
      value: s.effectiveness || s.score || 0
    }));

    return ASCIIChart.barChart(chartData, {
      title: 'Strategy Effectiveness Comparison',
      showValues: true
    });
  }

  async compareAgents(agentData = null) {
    const data = agentData || this.comparisons.agents;

    if (data.length === 0) {
      return 'No agent comparison data available';
    }

    const chartData = data.map(a => ({
      label: a.agent || a.id || a.name,
      value: a.performance || a.score || 0
    }));

    return ASCIIChart.barChart(chartData, {
      title: 'Agent Performance Comparison',
      showValues: true
    });
  }

  /**
   * Analytics
   */

  async predictConvergence(currentData = null) {
    const data = currentData || this.metrics.learningCurve;

    if (data.length < 5) {
      return 'Insufficient data for convergence prediction';
    }

    const recent = data.slice(-10);
    const avgImprovement = recent.reduce((sum, point, i) => {
      if (i === 0) return sum;
      return sum + (point.accuracy - recent[i - 1].accuracy);
    }, 0) / (recent.length - 1);

    const currentAccuracy = recent[recent.length - 1].accuracy;
    const targetAccuracy = this.options.targetAccuracy || 0.95;

    if (currentAccuracy >= targetAccuracy) {
      return `Target accuracy ${(targetAccuracy * 100).toFixed(1)}% already achieved!`;
    }

    if (avgImprovement <= 0) {
      return 'Warning: Learning has plateaued. Consider adjusting parameters.';
    }

    const episodesNeeded = Math.ceil((targetAccuracy - currentAccuracy) / avgImprovement);

    let output = '\nConvergence Prediction\n';
    output += '=====================\n\n';
    output += `Current Accuracy: ${(currentAccuracy * 100).toFixed(2)}%\n`;
    output += `Target Accuracy: ${(targetAccuracy * 100).toFixed(2)}%\n`;
    output += `Avg Improvement: ${(avgImprovement * 100).toFixed(4)}% per episode\n`;
    output += `Estimated Episodes to Target: ${episodesNeeded}\n`;

    return output;
  }

  async identifyBottlenecks(performanceData = null) {
    const data = performanceData || this.metrics;

    let output = '\nLearning Bottleneck Analysis\n';
    output += '============================\n\n';

    const bottlenecks = [];

    // Check learning rate
    if (this.metrics.learningCurve.length > 10) {
      const recent = this.metrics.learningCurve.slice(-10);
      const improvement = recent[recent.length - 1].accuracy - recent[0].accuracy;

      if (improvement < 0.01) {
        bottlenecks.push({
          type: 'Learning Plateau',
          severity: 'HIGH',
          description: 'Learning has plateaued. Consider increasing exploration or adjusting learning rate.'
        });
      }
    }

    // Check pattern diversity
    if (this.metrics.patternGrowth.length > 0) {
      const recentPatterns = this.metrics.patternGrowth.slice(-5);
      const patternGrowth = recentPatterns[recentPatterns.length - 1].count - recentPatterns[0].count;

      if (patternGrowth < 2) {
        bottlenecks.push({
          type: 'Low Pattern Diversity',
          severity: 'MEDIUM',
          description: 'Few new patterns discovered. Consider increasing exploration diversity.'
        });
      }
    }

    // Check agent skill variance
    if (this.metrics.agentSkills.size > 0) {
      const allSkills = Array.from(this.metrics.agentSkills.values());
      const avgSkillLevels = allSkills.map(skills => {
        const values = Object.values(skills);
        return values.reduce((sum, v) => sum + v, 0) / values.length;
      });

      const variance = avgSkillLevels.reduce((sum, v) => {
        const mean = avgSkillLevels.reduce((s, x) => s + x, 0) / avgSkillLevels.length;
        return sum + Math.pow(v - mean, 2);
      }, 0) / avgSkillLevels.length;

      if (variance < 0.01) {
        bottlenecks.push({
          type: 'Low Agent Diversity',
          severity: 'MEDIUM',
          description: 'Agents have similar skill levels. Consider role specialization.'
        });
      }
    }

    if (bottlenecks.length === 0) {
      output += 'No significant bottlenecks detected. Learning is progressing well.\n';
    } else {
      bottlenecks.forEach((b, i) => {
        output += `${i + 1}. [${b.severity}] ${b.type}\n`;
        output += `   ${b.description}\n\n`;
      });
    }

    return output;
  }

  async recommendOptimizations() {
    let output = '\nOptimization Recommendations\n';
    output += '============================\n\n';

    const recommendations = [];

    // Analyze learning curve
    if (this.metrics.learningCurve.length > 10) {
      const recent = this.metrics.learningCurve.slice(-10);
      const improvement = recent[recent.length - 1].accuracy - recent[0].accuracy;

      if (improvement < 0.01) {
        recommendations.push({
          category: 'Learning Rate',
          priority: 'HIGH',
          suggestion: 'Increase exploration rate or implement curriculum learning',
          expectedImpact: '+15-30% faster convergence'
        });
      } else if (improvement > 0.1) {
        recommendations.push({
          category: 'Learning Rate',
          priority: 'LOW',
          suggestion: 'Current learning rate is optimal. Maintain current settings.',
          expectedImpact: 'Stable performance'
        });
      }
    }

    // Topology optimization
    if (this.comparisons.topologies.length > 1) {
      const best = this.comparisons.topologies.reduce((max, t) =>
        t.accuracy > max.accuracy ? t : max
      );

      recommendations.push({
        category: 'Topology',
        priority: 'MEDIUM',
        suggestion: `${best.topology} topology shows best performance. Consider switching if not already using.`,
        expectedImpact: `+${((best.accuracy - 0.7) * 100).toFixed(0)}% accuracy improvement`
      });
    }

    // Agent specialization
    if (this.metrics.agentSkills.size > 2) {
      recommendations.push({
        category: 'Agent Roles',
        priority: 'MEDIUM',
        suggestion: 'Implement role-based task assignment to leverage agent specialization',
        expectedImpact: '+10-20% efficiency gain'
      });
    }

    // Pattern reuse
    if (this.metrics.knowledgeGraph.nodes.length > 10) {
      recommendations.push({
        category: 'Knowledge Transfer',
        priority: 'LOW',
        suggestion: 'Enable pattern transfer between similar tasks for faster learning',
        expectedImpact: '+20-40% reduction in training time'
      });
    }

    if (recommendations.length === 0) {
      output += 'System is operating optimally. No immediate optimizations needed.\n';
    } else {
      recommendations.sort((a, b) => {
        const priority = { HIGH: 3, MEDIUM: 2, LOW: 1 };
        return priority[b.priority] - priority[a.priority];
      });

      recommendations.forEach((r, i) => {
        output += `${i + 1}. [${r.priority}] ${r.category}\n`;
        output += `   ${r.suggestion}\n`;
        output += `   Expected Impact: ${r.expectedImpact}\n\n`;
      });
    }

    return output;
  }

  /**
   * Data Management
   */

  updateMetrics(newData) {
    if (newData.learningCurve) {
      this.metrics.learningCurve.push(...newData.learningCurve);
      if (this.metrics.learningCurve.length > this.options.historySize) {
        this.metrics.learningCurve = this.metrics.learningCurve.slice(-this.options.historySize);
      }
    }

    if (newData.decisionQuality) {
      this.metrics.decisionQuality.push(...newData.decisionQuality);
      if (this.metrics.decisionQuality.length > this.options.historySize) {
        this.metrics.decisionQuality = this.metrics.decisionQuality.slice(-this.options.historySize);
      }
    }

    if (newData.patternGrowth) {
      this.metrics.patternGrowth.push(...newData.patternGrowth);
      if (this.metrics.patternGrowth.length > this.options.historySize) {
        this.metrics.patternGrowth = this.metrics.patternGrowth.slice(-this.options.historySize);
      }
    }

    if (newData.agentSkills) {
      Object.entries(newData.agentSkills).forEach(([agentId, skills]) => {
        this.metrics.agentSkills.set(agentId, skills);
      });
    }

    if (newData.knowledgeGraph) {
      this.metrics.knowledgeGraph = newData.knowledgeGraph;
    }

    if (newData.topologies) {
      this.comparisons.topologies = newData.topologies;
    }

    if (newData.strategies) {
      this.comparisons.strategies = newData.strategies;
    }

    if (newData.agents) {
      this.comparisons.agents = newData.agents;
    }
  }

  /**
   * Export Functionality
   */

  async exportHTML(outputPath) {
    const html = await this.generateHTML();
    await fs.writeFile(outputPath, html, 'utf8');
    return outputPath;
  }

  async exportMarkdown(outputPath) {
    const markdown = await this.generateMarkdown();
    await fs.writeFile(outputPath, markdown, 'utf8');
    return outputPath;
  }

  async exportJSON(outputPath) {
    const data = {
      timestamp: new Date().toISOString(),
      metrics: {
        learningCurve: this.metrics.learningCurve,
        decisionQuality: this.metrics.decisionQuality,
        patternGrowth: this.metrics.patternGrowth,
        agentSkills: Object.fromEntries(this.metrics.agentSkills),
        knowledgeGraph: this.metrics.knowledgeGraph
      },
      comparisons: this.comparisons,
      analytics: {
        convergencePrediction: await this.predictConvergence(),
        bottlenecks: await this.identifyBottlenecks(),
        recommendations: await this.recommendOptimizations()
      }
    };

    await fs.writeFile(outputPath, JSON.stringify(data, null, 2), 'utf8');
    return outputPath;
  }

  async generateHTML() {
    const timestamp = new Date().toISOString();

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ReasoningBank Learning Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            opacity: 0.9;
            font-size: 1.1em;
        }
        .dashboard {
            padding: 30px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .chart-container {
            position: relative;
            height: 300px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            opacity: 0.9;
            font-size: 0.9em;
        }
        .analysis {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin-top: 20px;
            border-radius: 4px;
        }
        .analysis h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        .recommendation {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 3px solid #28a745;
        }
        .recommendation.high { border-left-color: #dc3545; }
        .recommendation.medium { border-left-color: #ffc107; }
        .recommendation.low { border-left-color: #28a745; }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 ReasoningBank Learning Dashboard</h1>
            <p>Real-time E2B Swarm Learning Analytics</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: ${timestamp}</p>
        </div>

        <div class="dashboard">
            <!-- Key Statistics -->
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">${this.metrics.learningCurve.length > 0 ? (this.metrics.learningCurve[this.metrics.learningCurve.length - 1].accuracy * 100).toFixed(1) : 0}%</div>
                    <div class="stat-label">Current Accuracy</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${this.metrics.learningCurve.length}</div>
                    <div class="stat-label">Episodes Completed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${this.metrics.patternGrowth.length > 0 ? this.metrics.patternGrowth[this.metrics.patternGrowth.length - 1].count : 0}</div>
                    <div class="stat-label">Patterns Discovered</div>
                </div>
                <div class="stat">
                    <div class="stat-value">${this.metrics.agentSkills.size}</div>
                    <div class="stat-label">Active Agents</div>
                </div>
            </div>

            <!-- Charts -->
            <div class="grid">
                <div class="card">
                    <h2>📈 Learning Curve</h2>
                    <div class="chart-container">
                        <canvas id="learningCurve"></canvas>
                    </div>
                </div>

                <div class="card">
                    <h2>⭐ Decision Quality</h2>
                    <div class="chart-container">
                        <canvas id="decisionQuality"></canvas>
                    </div>
                </div>

                <div class="card">
                    <h2>🔍 Pattern Discovery</h2>
                    <div class="chart-container">
                        <canvas id="patternGrowth"></canvas>
                    </div>
                </div>

                <div class="card">
                    <h2>🤖 Agent Performance</h2>
                    <div class="chart-container">
                        <canvas id="agentComparison"></canvas>
                    </div>
                </div>
            </div>

            <!-- Analytics -->
            <div class="analysis">
                <h3>🎯 Convergence Prediction</h3>
                <pre>${await this.predictConvergence()}</pre>
            </div>

            <div class="analysis">
                <h3>🔧 Bottleneck Analysis</h3>
                <pre>${await this.identifyBottlenecks()}</pre>
            </div>

            <div class="analysis">
                <h3>💡 Optimization Recommendations</h3>
                <pre>${await this.recommendOptimizations()}</pre>
            </div>
        </div>

        <div class="footer">
            ReasoningBank E2B Swarm Learning Dashboard | Generated by Neural Trader
        </div>
    </div>

    <script>
        // Learning Curve Chart
        new Chart(document.getElementById('learningCurve'), {
            type: 'line',
            data: {
                labels: ${JSON.stringify(this.metrics.learningCurve.map((_, i) => i + 1))},
                datasets: [{
                    label: 'Accuracy',
                    data: ${JSON.stringify(this.metrics.learningCurve.map(d => d.accuracy * 100))},
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: 'Accuracy (%)' }
                    },
                    x: {
                        title: { display: true, text: 'Episodes' }
                    }
                }
            }
        });

        // Decision Quality Chart
        new Chart(document.getElementById('decisionQuality'), {
            type: 'line',
            data: {
                labels: ${JSON.stringify(this.metrics.decisionQuality.map((_, i) => i + 1))},
                datasets: [{
                    label: 'Quality Score',
                    data: ${JSON.stringify(this.metrics.decisionQuality.map(d => d.score || d.value))},
                    borderColor: '#764ba2',
                    backgroundColor: 'rgba(118, 75, 162, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Quality Score' }
                    },
                    x: {
                        title: { display: true, text: 'Time' }
                    }
                }
            }
        });

        // Pattern Growth Chart
        new Chart(document.getElementById('patternGrowth'), {
            type: 'bar',
            data: {
                labels: ${JSON.stringify(this.metrics.patternGrowth.map((_, i) => i + 1))},
                datasets: [{
                    label: 'Patterns',
                    data: ${JSON.stringify(this.metrics.patternGrowth.map(d => d.count))},
                    backgroundColor: 'rgba(102, 126, 234, 0.8)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Pattern Count' }
                    },
                    x: {
                        title: { display: true, text: 'Episodes' }
                    }
                }
            }
        });

        // Agent Comparison Chart
        new Chart(document.getElementById('agentComparison'), {
            type: 'radar',
            data: {
                labels: ${JSON.stringify(this.comparisons.agents.map(a => a.agent || a.id || 'Agent'))},
                datasets: [{
                    label: 'Performance',
                    data: ${JSON.stringify(this.comparisons.agents.map(a => a.performance || a.score || 0))},
                    backgroundColor: 'rgba(118, 75, 162, 0.2)',
                    borderColor: '#764ba2',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    r: {
                        beginAtZero: true
                    }
                }
            }
        });
    </script>
</body>
</html>`;
  }

  async generateMarkdown() {
    let md = '# ReasoningBank E2B Swarm Learning Dashboard\n\n';
    md += `**Generated:** ${new Date().toISOString()}\n\n`;

    md += '## 📊 Key Metrics\n\n';
    md += `- **Current Accuracy:** ${this.metrics.learningCurve.length > 0 ? (this.metrics.learningCurve[this.metrics.learningCurve.length - 1].accuracy * 100).toFixed(2) : 0}%\n`;
    md += `- **Episodes Completed:** ${this.metrics.learningCurve.length}\n`;
    md += `- **Patterns Discovered:** ${this.metrics.patternGrowth.length > 0 ? this.metrics.patternGrowth[this.metrics.patternGrowth.length - 1].count : 0}\n`;
    md += `- **Active Agents:** ${this.metrics.agentSkills.size}\n\n`;

    md += '## 📈 Learning Curve\n\n';
    md += '```\n' + await this.displayLearningCurve() + '\n```\n\n';

    md += '## ⭐ Decision Quality\n\n';
    md += '```\n' + await this.displayDecisionQuality() + '\n```\n\n';

    md += '## 🔍 Pattern Growth\n\n';
    md += '```\n' + await this.displayPatternGrowth() + '\n```\n\n';

    md += '## 🎯 Analytics\n\n';
    md += '### Convergence Prediction\n\n';
    md += '```\n' + await this.predictConvergence() + '\n```\n\n';

    md += '### Bottleneck Analysis\n\n';
    md += '```\n' + await this.identifyBottlenecks() + '\n```\n\n';

    md += '### Optimization Recommendations\n\n';
    md += '```\n' + await this.recommendOptimizations() + '\n```\n\n';

    return md;
  }

  /**
   * Generate complete dashboard report
   */
  async generateCompleteReport() {
    let report = '\n';
    report += '═══════════════════════════════════════════════════════════════\n';
    report += '         ReasoningBank E2B Swarm Learning Dashboard           \n';
    report += '═══════════════════════════════════════════════════════════════\n\n';

    // Display all visualizations
    report += await this.displayLearningCurve() + '\n\n';
    report += await this.displayDecisionQuality() + '\n\n';
    report += await this.displayPatternGrowth() + '\n\n';

    if (this.metrics.agentSkills.size > 0) {
      report += await this.displayAgentSkills() + '\n\n';
    }

    if (this.metrics.knowledgeGraph.nodes.length > 0) {
      report += await this.displayKnowledgeGraph() + '\n\n';
    }

    // Comparisons
    if (this.comparisons.topologies.length > 0) {
      report += await this.compareTopologies() + '\n\n';
    }

    if (this.comparisons.strategies.length > 0) {
      report += await this.compareStrategies() + '\n\n';
    }

    if (this.comparisons.agents.length > 0) {
      report += await this.compareAgents() + '\n\n';
    }

    // Analytics
    report += await this.predictConvergence() + '\n\n';
    report += await this.identifyBottlenecks() + '\n\n';
    report += await this.recommendOptimizations() + '\n\n';

    report += '═══════════════════════════════════════════════════════════════\n';
    report += `Generated: ${new Date().toISOString()}\n`;
    report += '═══════════════════════════════════════════════════════════════\n';

    return report;
  }
}

module.exports = {
  LearningDashboard,
  ASCIIChart
};
