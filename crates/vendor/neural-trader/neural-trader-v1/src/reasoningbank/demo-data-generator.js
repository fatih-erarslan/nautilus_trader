/**
 * Demo Learning Data Generator
 * Generates realistic learning data for dashboard testing
 */

const fs = require('fs').promises;
const path = require('path');

class DemoDataGenerator {
  constructor(options = {}) {
    this.options = {
      episodes: 100,
      agents: 5,
      patterns: 20,
      topologies: ['mesh', 'hierarchical', 'ring', 'star'],
      strategies: ['q-learning', 'policy-gradient', 'actor-critic', 'ppo'],
      ...options
    };
  }

  /**
   * Generate complete learning dataset
   */
  generate() {
    return {
      metadata: {
        generated: new Date().toISOString(),
        episodes: this.options.episodes,
        agents: this.options.agents,
        version: '1.0.0'
      },
      learningCurve: this.generateLearningCurve(),
      decisionQuality: this.generateDecisionQuality(),
      patternGrowth: this.generatePatternGrowth(),
      agentSkills: this.generateAgentSkills(),
      knowledgeGraph: this.generateKnowledgeGraph(),
      topologies: this.generateTopologyComparison(),
      strategies: this.generateStrategyComparison(),
      agents: this.generateAgentComparison()
    };
  }

  /**
   * Generate learning curve with realistic progression
   */
  generateLearningCurve() {
    const curve = [];
    let accuracy = 0.25 + Math.random() * 0.1; // Start 25-35%
    const targetAccuracy = 0.88 + Math.random() * 0.08; // Target 88-96%
    const learningRate = 0.015 + Math.random() * 0.01;

    for (let i = 0; i < this.options.episodes; i++) {
      // Sigmoid-like learning curve with noise
      const progress = i / this.options.episodes;
      const sigmoid = 1 / (1 + Math.exp(-10 * (progress - 0.5)));
      const improvement = learningRate * sigmoid;
      const noise = (Math.random() - 0.5) * 0.02;

      accuracy = Math.min(
        targetAccuracy,
        accuracy + improvement + noise
      );

      // Occasional plateaus
      if (Math.random() < 0.1) {
        accuracy += (Math.random() - 0.5) * 0.005;
      }

      curve.push({
        episode: i + 1,
        accuracy: Math.max(0, Math.min(1, accuracy)),
        loss: Math.max(0, 1 - accuracy + (Math.random() - 0.5) * 0.1),
        timestamp: new Date(Date.now() - (this.options.episodes - i) * 300000).toISOString()
      });
    }

    return curve;
  }

  /**
   * Generate decision quality metrics
   */
  generateDecisionQuality() {
    const quality = [];
    let baseQuality = 0.5;

    for (let i = 0; i < this.options.episodes * 2; i++) {
      // Quality improves with episodes but has variance
      const episodeProgress = i / (this.options.episodes * 2);
      baseQuality = 0.5 + (episodeProgress * 0.4);
      const variance = (Math.random() - 0.5) * 0.15;

      quality.push({
        score: Math.max(0, Math.min(1, baseQuality + variance)),
        confidence: 0.6 + episodeProgress * 0.3,
        timestamp: new Date(Date.now() - (this.options.episodes * 2 - i) * 150000).toISOString()
      });
    }

    return quality;
  }

  /**
   * Generate pattern discovery progression
   */
  generatePatternGrowth() {
    const growth = [];
    let totalPatterns = 0;
    let discoveryRate = 0.5;

    for (let i = 0; i < this.options.episodes; i++) {
      // Discovery rate decreases over time (diminishing returns)
      discoveryRate = 0.5 * Math.exp(-i / 30);

      if (Math.random() < discoveryRate) {
        totalPatterns += Math.floor(Math.random() * 3) + 1;
      }

      growth.push({
        count: totalPatterns,
        newPatterns: i > 0 ? totalPatterns - growth[i - 1].count : totalPatterns,
        discoveryRate: discoveryRate,
        timestamp: new Date(Date.now() - (this.options.episodes - i) * 300000).toISOString()
      });
    }

    return growth;
  }

  /**
   * Generate agent skill matrices
   */
  generateAgentSkills() {
    const skills = {};
    const skillTypes = [
      'pattern-recognition',
      'decision-making',
      'exploration',
      'exploitation',
      'coordination',
      'adaptation'
    ];

    for (let i = 0; i < this.options.agents; i++) {
      const agentId = `agent-${String(i + 1).padStart(3, '0')}`;
      skills[agentId] = {};

      // Each agent has different strengths
      skillTypes.forEach(skill => {
        const base = 0.5 + Math.random() * 0.3;
        skills[agentId][skill] = Math.min(0.95, base + (Math.random() - 0.3) * 0.2);
      });
    }

    return skills;
  }

  /**
   * Generate knowledge graph
   */
  generateKnowledgeGraph() {
    const patterns = [
      'Volatility spike → Risk reduction',
      'Low liquidity → Position sizing',
      'Trend reversal → Exit strategy',
      'Correlation breakdown → Hedge adjustment',
      'News sentiment → Entry timing',
      'Volume spike → Momentum confirmation',
      'Support level → Buy opportunity',
      'Resistance level → Profit taking',
      'Moving average cross → Trend change',
      'RSI divergence → Reversal signal',
      'High spread → Market inefficiency',
      'Order imbalance → Price pressure',
      'Sector rotation → Rebalancing',
      'Earnings surprise → Volatility expansion',
      'Central bank action → Risk-on/off'
    ];

    const nodes = patterns.slice(0, this.options.patterns).map((pattern, i) => ({
      id: `p${i + 1}`,
      pattern,
      confidence: 0.65 + Math.random() * 0.3,
      usageCount: Math.floor(Math.random() * 50) + 10,
      successRate: 0.6 + Math.random() * 0.3
    }));

    const edges = [];
    for (let i = 0; i < nodes.length; i++) {
      const numConnections = Math.floor(Math.random() * 3) + 1;
      for (let j = 0; j < numConnections; j++) {
        const targetIdx = Math.floor(Math.random() * nodes.length);
        if (targetIdx !== i) {
          edges.push({
            from: nodes[i].id,
            to: nodes[targetIdx].id,
            weight: Math.random() * 0.8 + 0.2,
            type: Math.random() < 0.5 ? 'implies' : 'correlates'
          });
        }
      }
    }

    return { nodes, edges };
  }

  /**
   * Generate topology comparison data
   */
  generateTopologyComparison() {
    return this.options.topologies.map(topology => {
      const baseAccuracy = 0.75 + Math.random() * 0.15;
      const episodes = Math.floor(40 + Math.random() * 20);

      return {
        topology: topology.charAt(0).toUpperCase() + topology.slice(1),
        accuracy: baseAccuracy,
        episodes: episodes,
        convergenceSpeed: 1 / episodes,
        communicationOverhead: Math.random() * 0.3,
        scalability: Math.random() * 0.8 + 0.2
      };
    });
  }

  /**
   * Generate strategy comparison data
   */
  generateStrategyComparison() {
    return this.options.strategies.map(strategy => {
      const effectiveness = 0.75 + Math.random() * 0.2;
      const convergence = Math.floor(30 + Math.random() * 25);

      return {
        strategy: strategy.split('-').map(w =>
          w.charAt(0).toUpperCase() + w.slice(1)
        ).join(' '),
        effectiveness,
        convergence,
        stability: Math.random() * 0.3 + 0.6,
        sampleEfficiency: Math.random() * 0.5 + 0.5
      };
    });
  }

  /**
   * Generate agent comparison data
   */
  generateAgentComparison() {
    const comparisons = [];

    for (let i = 0; i < this.options.agents; i++) {
      const performance = 0.7 + Math.random() * 0.25;
      const patterns = Math.floor(Math.random() * 15) + 5;

      comparisons.push({
        agent: `Agent-${String(i + 1).padStart(3, '0')}`,
        id: `agent-${String(i + 1).padStart(3, '0')}`,
        performance,
        patterns,
        decisions: Math.floor(Math.random() * 500) + 100,
        successRate: 0.6 + Math.random() * 0.3
      });
    }

    return comparisons;
  }

  /**
   * Save generated data to file
   */
  async save(outputPath) {
    const data = this.generate();
    await fs.writeFile(outputPath, JSON.stringify(data, null, 2), 'utf8');
    return outputPath;
  }

  /**
   * Generate and save multiple datasets
   */
  async generateMultiple(count, outputDir) {
    await fs.mkdir(outputDir, { recursive: true });

    const files = [];
    for (let i = 0; i < count; i++) {
      const filename = `learning-data-${i + 1}.json`;
      const filepath = path.join(outputDir, filename);
      await this.save(filepath);
      files.push(filepath);
    }

    return files;
  }
}

// CLI usage
if (require.main === module) {
  const args = process.argv.slice(2);
  const outputPath = args[0] || path.join(
    process.cwd(),
    'docs',
    'reasoningbank',
    'demo-data.json'
  );

  const generator = new DemoDataGenerator({
    episodes: 100,
    agents: 5,
    patterns: 15
  });

  generator.save(outputPath).then(path => {
    console.log(`✓ Demo data generated: ${path}`);
  }).catch(error => {
    console.error('✗ Failed to generate demo data:', error.message);
    process.exit(1);
  });
}

module.exports = { DemoDataGenerator };
