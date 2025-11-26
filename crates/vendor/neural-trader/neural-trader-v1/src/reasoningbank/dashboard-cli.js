#!/usr/bin/env node

/**
 * ReasoningBank Learning Dashboard CLI Integration
 * Adds learning dashboard commands to e2b-swarm-cli
 */

const { LearningDashboard } = require('./learning-dashboard');
const fs = require('fs').promises;
const path = require('path');
const chalk = require('chalk');

/**
 * Dashboard CLI Commands
 */
class DashboardCLI {
  constructor(stateManager, formatter) {
    this.stateManager = stateManager;
    this.formatter = formatter;
    this.dashboard = new LearningDashboard({
      updateInterval: 1000,
      historySize: 100,
      targetAccuracy: 0.95
    });

    this.dashboardDir = path.join(process.cwd(), 'docs', 'reasoningbank', 'dashboards');
    this.reportsDir = path.join(process.cwd(), 'docs', 'reasoningbank', 'reports');
  }

  async ensureDirectories() {
    await fs.mkdir(this.dashboardDir, { recursive: true });
    await fs.mkdir(this.reportsDir, { recursive: true });
  }

  /**
   * Load learning data from file or state
   */
  async loadLearningData(source = null) {
    if (source) {
      try {
        const data = await fs.readFile(source, 'utf8');
        return JSON.parse(data);
      } catch (error) {
        this.formatter.error(`Failed to load data from ${source}: ${error.message}`);
        return null;
      }
    }

    // Load from state or generate demo data
    return this.generateDemoData();
  }

  /**
   * Generate demo data for visualization
   */
  generateDemoData() {
    const episodes = 50;
    const learningCurve = [];
    const decisionQuality = [];
    const patternGrowth = [];

    let accuracy = 0.3;
    let patterns = 0;

    for (let i = 0; i < episodes; i++) {
      // Simulate learning curve with some variance
      accuracy += (Math.random() * 0.02) + 0.005;
      accuracy = Math.min(accuracy, 0.95);

      learningCurve.push({
        episode: i + 1,
        accuracy: accuracy,
        timestamp: new Date(Date.now() - (episodes - i) * 60000).toISOString()
      });

      // Decision quality improves with accuracy
      const quality = 0.5 + (accuracy * 0.5) + (Math.random() * 0.1 - 0.05);
      decisionQuality.push({
        score: Math.max(0, Math.min(1, quality)),
        timestamp: new Date(Date.now() - (episodes - i) * 60000).toISOString()
      });

      // Pattern discovery slows over time
      if (Math.random() < (0.3 - i * 0.005)) {
        patterns += Math.floor(Math.random() * 3) + 1;
      }
      patternGrowth.push({
        count: patterns,
        timestamp: new Date(Date.now() - (episodes - i) * 60000).toISOString()
      });
    }

    // Agent skills
    const agentSkills = {
      'agent-001': {
        'pattern-recognition': 0.85,
        'decision-making': 0.78,
        'exploration': 0.65,
        'exploitation': 0.92,
        'coordination': 0.70
      },
      'agent-002': {
        'pattern-recognition': 0.72,
        'decision-making': 0.88,
        'exploration': 0.80,
        'exploitation': 0.75,
        'coordination': 0.82
      },
      'agent-003': {
        'pattern-recognition': 0.90,
        'decision-making': 0.70,
        'exploration': 0.95,
        'exploitation': 0.68,
        'coordination': 0.75
      }
    };

    // Knowledge graph
    const knowledgeGraph = {
      nodes: [
        { id: 'p1', pattern: 'High volatility → Risk reduction', confidence: 0.92 },
        { id: 'p2', pattern: 'Low liquidity → Position sizing', confidence: 0.88 },
        { id: 'p3', pattern: 'Trend reversal → Exit strategy', confidence: 0.85 },
        { id: 'p4', pattern: 'Correlation breakdown → Hedge adjustment', confidence: 0.79 },
        { id: 'p5', pattern: 'News sentiment → Entry timing', confidence: 0.75 }
      ],
      edges: [
        { from: 'p1', to: 'p2', weight: 0.65 },
        { from: 'p1', to: 'p3', weight: 0.72 },
        { from: 'p2', to: 'p4', weight: 0.58 },
        { from: 'p3', to: 'p5', weight: 0.81 },
        { from: 'p4', to: 'p5', weight: 0.45 }
      ]
    };

    // Topology comparisons
    const topologies = [
      { topology: 'Mesh', accuracy: 0.89, episodes: 45 },
      { topology: 'Hierarchical', accuracy: 0.85, episodes: 50 },
      { topology: 'Ring', accuracy: 0.82, episodes: 55 },
      { topology: 'Star', accuracy: 0.87, episodes: 48 }
    ];

    // Strategy comparisons
    const strategies = [
      { strategy: 'Q-Learning', effectiveness: 0.84, convergence: 42 },
      { strategy: 'Policy Gradient', effectiveness: 0.88, convergence: 38 },
      { strategy: 'Actor-Critic', effectiveness: 0.91, convergence: 35 },
      { strategy: 'PPO', effectiveness: 0.86, convergence: 40 }
    ];

    // Agent comparisons
    const agents = [
      { agent: 'Agent-001', performance: 0.87, patterns: 15 },
      { agent: 'Agent-002', performance: 0.82, patterns: 12 },
      { agent: 'Agent-003', performance: 0.91, patterns: 18 }
    ];

    return {
      learningCurve,
      decisionQuality,
      patternGrowth,
      agentSkills,
      knowledgeGraph,
      topologies,
      strategies,
      agents
    };
  }

  /**
   * Display live dashboard in terminal
   */
  async displayLive(options = {}) {
    const { interval = 2000, duration = 60000 } = options;

    this.formatter.banner('ReasoningBank Learning Dashboard - Live Mode');

    const data = await this.loadLearningData();
    this.dashboard.updateMetrics(data);

    const startTime = Date.now();
    let iteration = 0;

    const updateDisplay = async () => {
      // Clear screen
      console.clear();

      this.formatter.banner('ReasoningBank Learning Dashboard');
      console.log(chalk.gray(`Live Update ${iteration + 1} | ${new Date().toLocaleTimeString()}\n`));

      // Display all visualizations
      console.log(await this.dashboard.displayLearningCurve());
      console.log('\n' + await this.dashboard.displayDecisionQuality());
      console.log('\n' + await this.dashboard.displayPatternGrowth());

      if (this.dashboard.metrics.agentSkills.size > 0) {
        console.log('\n' + await this.dashboard.displayAgentSkills());
      }

      console.log('\n' + await this.dashboard.compareTopologies());
      console.log('\n' + await this.dashboard.predictConvergence());

      iteration++;

      if (Date.now() - startTime < duration) {
        setTimeout(updateDisplay, interval);
      } else {
        this.formatter.success('\nLive dashboard session ended');
      }
    };

    await updateDisplay();
  }

  /**
   * Generate and save HTML dashboard
   */
  async generateHTML(options = {}) {
    const { output, open = false } = options;

    await this.ensureDirectories();

    this.formatter.info('Loading learning data...');
    const data = await this.loadLearningData(options.source);

    if (!data) {
      this.formatter.error('No data available for dashboard generation');
      return;
    }

    this.dashboard.updateMetrics(data);

    this.formatter.info('Generating HTML dashboard...');

    const outputPath = output || path.join(
      this.dashboardDir,
      `dashboard-${Date.now()}.html`
    );

    await this.dashboard.exportHTML(outputPath);

    this.formatter.success(`HTML dashboard saved to: ${outputPath}`);

    if (open) {
      this.formatter.info('Opening dashboard in browser...');
      const opener = require('opener');
      opener(outputPath);
    }

    return outputPath;
  }

  /**
   * Generate and save Markdown report
   */
  async generateReport(options = {}) {
    const { format = 'markdown', output } = options;

    await this.ensureDirectories();

    this.formatter.info('Loading learning data...');
    const data = await this.loadLearningData(options.source);

    if (!data) {
      this.formatter.error('No data available for report generation');
      return;
    }

    this.dashboard.updateMetrics(data);

    this.formatter.info(`Generating ${format.toUpperCase()} report...`);

    let outputPath;
    switch (format) {
      case 'html':
        outputPath = output || path.join(this.reportsDir, `report-${Date.now()}.html`);
        await this.dashboard.exportHTML(outputPath);
        break;

      case 'json':
        outputPath = output || path.join(this.reportsDir, `report-${Date.now()}.json`);
        await this.dashboard.exportJSON(outputPath);
        break;

      case 'markdown':
      default:
        outputPath = output || path.join(this.reportsDir, `report-${Date.now()}.md`);
        await this.dashboard.exportMarkdown(outputPath);
        break;
    }

    this.formatter.success(`Report saved to: ${outputPath}`);
    return outputPath;
  }

  /**
   * Show stats for specific agent
   */
  async showAgentStats(options = {}) {
    const { agent } = options;

    this.formatter.banner(`Agent Statistics: ${agent || 'All Agents'}`);

    const data = await this.loadLearningData(options.source);

    if (!data) {
      this.formatter.error('No data available');
      return;
    }

    this.dashboard.updateMetrics(data);

    // Display agent skills
    console.log(await this.dashboard.displayAgentSkills(agent));

    // Display agent comparison if showing all
    if (!agent) {
      console.log('\n' + await this.dashboard.compareAgents());
    }

    // Display knowledge graph
    console.log('\n' + await this.dashboard.displayKnowledgeGraph());
  }

  /**
   * Display analytics
   */
  async showAnalytics(options = {}) {
    this.formatter.banner('Learning Analytics');

    const data = await this.loadLearningData(options.source);

    if (!data) {
      this.formatter.error('No data available');
      return;
    }

    this.dashboard.updateMetrics(data);

    console.log(await this.dashboard.predictConvergence());
    console.log('\n' + await this.dashboard.identifyBottlenecks());
    console.log('\n' + await this.dashboard.recommendOptimizations());
  }

  /**
   * Export learning data
   */
  async exportData(options = {}) {
    const { format = 'json', output } = options;

    await this.ensureDirectories();

    this.formatter.info('Collecting learning data...');
    const data = await this.loadLearningData(options.source);

    if (!data) {
      this.formatter.error('No data available for export');
      return;
    }

    const outputPath = output || path.join(
      this.reportsDir,
      `learning-data-${Date.now()}.${format}`
    );

    if (format === 'json') {
      await fs.writeFile(outputPath, JSON.stringify(data, null, 2), 'utf8');
    } else {
      this.formatter.error(`Unsupported format: ${format}`);
      return;
    }

    this.formatter.success(`Data exported to: ${outputPath}`);
    return outputPath;
  }

  /**
   * Quick stats display
   */
  async quickStats(options = {}) {
    const data = await this.loadLearningData(options.source);

    if (!data) {
      this.formatter.error('No data available');
      return;
    }

    this.dashboard.updateMetrics(data);

    const stats = {
      episodes: this.dashboard.metrics.learningCurve.length,
      currentAccuracy: this.dashboard.metrics.learningCurve.length > 0
        ? (this.dashboard.metrics.learningCurve[this.dashboard.metrics.learningCurve.length - 1].accuracy * 100).toFixed(2)
        : 0,
      patterns: this.dashboard.metrics.patternGrowth.length > 0
        ? this.dashboard.metrics.patternGrowth[this.dashboard.metrics.patternGrowth.length - 1].count
        : 0,
      agents: this.dashboard.metrics.agentSkills.size,
      topTopology: this.dashboard.comparisons.topologies.length > 0
        ? this.dashboard.comparisons.topologies.reduce((max, t) => t.accuracy > max.accuracy ? t : max).topology
        : 'N/A'
    };

    if (this.formatter.jsonMode) {
      this.formatter.json(stats);
    } else {
      console.log('\n' + chalk.bold('Quick Stats'));
      console.log(chalk.gray('═'.repeat(40)));
      console.log(chalk.cyan('Episodes:       ') + chalk.white(stats.episodes));
      console.log(chalk.cyan('Accuracy:       ') + chalk.white(stats.currentAccuracy + '%'));
      console.log(chalk.cyan('Patterns:       ') + chalk.white(stats.patterns));
      console.log(chalk.cyan('Active Agents:  ') + chalk.white(stats.agents));
      console.log(chalk.cyan('Top Topology:   ') + chalk.white(stats.topTopology));
      console.log(chalk.gray('═'.repeat(40)) + '\n');
    }

    return stats;
  }
}

module.exports = { DashboardCLI };
