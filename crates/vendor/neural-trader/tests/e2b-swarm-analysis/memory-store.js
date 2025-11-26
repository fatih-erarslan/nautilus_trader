#!/usr/bin/env node

/**
 * Memory Storage for E2B Swarm Analysis
 *
 * Stores benchmark results in claude-flow memory
 * for cross-session persistence and analysis
 *
 * @module tests/e2b-swarm-analysis/memory-store
 */

const fs = require('fs').promises;
const path = require('path');
const { execSync } = require('child_process');

/**
 * Memory Store Manager
 */
class MemoryStoreManager {
  constructor(namespace = 'analysis/e2b-swarm') {
    this.namespace = namespace;
  }

  async storeMetrics(metricsData) {
    console.log(`\n📦 Storing metrics in memory (${this.namespace})...\n`);

    const keys = [
      'topology-performance',
      'scaling-metrics',
      'reasoningbank-integration',
      'reliability-results',
      'communication-overhead',
      'summary'
    ];

    const results = {
      stored: [],
      failed: []
    };

    try {
      // Store topology performance
      if (metricsData.topology) {
        await this.storeKey('topology-performance', metricsData.topology);
        results.stored.push('topology-performance');
      }

      // Store scaling metrics
      if (metricsData.scaling) {
        await this.storeKey('scaling-metrics', metricsData.scaling);
        results.stored.push('scaling-metrics');
      }

      // Store ReasoningBank data
      if (metricsData.reasoningBank) {
        await this.storeKey('reasoningbank-integration', metricsData.reasoningBank);
        results.stored.push('reasoningbank-integration');
      }

      // Store reliability results
      if (metricsData.reliability) {
        await this.storeKey('reliability-results', metricsData.reliability);
        results.stored.push('reliability-results');
      }

      // Store communication metrics
      if (metricsData.communication) {
        await this.storeKey('communication-overhead', metricsData.communication);
        results.stored.push('communication-overhead');
      }

      // Store summary
      const summary = {
        totalOperations: metricsData.operations?.length || 0,
        successRate: this.calculateSuccessRate(metricsData.operations || []),
        timestamp: new Date().toISOString(),
        topologiesTested: Object.keys(metricsData.topology || {}),
        agentScalesTested: this.extractAgentScales(metricsData)
      };

      await this.storeKey('summary', summary);
      results.stored.push('summary');

      // Store operations log
      if (metricsData.operations && metricsData.operations.length > 0) {
        await this.storeKey('operations-log', {
          count: metricsData.operations.length,
          operations: metricsData.operations.slice(0, 100) // Store first 100
        });
        results.stored.push('operations-log');
      }

      console.log(`✅ Successfully stored ${results.stored.length} keys:`);
      results.stored.forEach(key => console.log(`   - ${this.namespace}/${key}`));

    } catch (error) {
      console.error(`❌ Error storing metrics: ${error.message}`);
      results.failed.push(error.message);
    }

    return results;
  }

  async storeKey(key, data) {
    const fullKey = `${this.namespace}/${key}`;

    try {
      // Use claude-flow memory CLI
      const jsonData = JSON.stringify(data);
      const command = `npx claude-flow@alpha hooks memory-store --key "${fullKey}" --value '${jsonData.replace(/'/g, "'\\''")}' --ttl 604800`; // 7 days TTL

      execSync(command, { stdio: 'pipe', encoding: 'utf8' });
      console.log(`  ✓ Stored ${fullKey}`);

    } catch (error) {
      // Fallback: store to local file
      console.warn(`  ⚠️  Memory store failed for ${fullKey}, using local cache`);

      const cacheDir = path.join(__dirname, '.memory-cache');
      await fs.mkdir(cacheDir, { recursive: true });

      const cacheFile = path.join(cacheDir, `${key}.json`);
      await fs.writeFile(cacheFile, JSON.stringify(data, null, 2));
    }
  }

  async retrieveKey(key) {
    const fullKey = `${this.namespace}/${key}`;

    try {
      const command = `npx claude-flow@alpha hooks memory-retrieve --key "${fullKey}"`;
      const output = execSync(command, { stdio: 'pipe', encoding: 'utf8' });

      return JSON.parse(output);

    } catch (error) {
      // Try local cache
      const cacheFile = path.join(__dirname, '.memory-cache', `${key}.json`);

      try {
        const data = await fs.readFile(cacheFile, 'utf8');
        return JSON.parse(data);
      } catch {
        return null;
      }
    }
  }

  async listStoredKeys() {
    try {
      const command = `npx claude-flow@alpha hooks memory-list --namespace "${this.namespace}"`;
      const output = execSync(command, { stdio: 'pipe', encoding: 'utf8' });

      return JSON.parse(output);

    } catch (error) {
      // Try local cache
      const cacheDir = path.join(__dirname, '.memory-cache');

      try {
        const files = await fs.readdir(cacheDir);
        return files.map(f => f.replace('.json', ''));
      } catch {
        return [];
      }
    }
  }

  calculateSuccessRate(operations) {
    if (operations.length === 0) return 0;

    const successCount = operations.filter(op => op.success).length;
    return (successCount / operations.length) * 100;
  }

  extractAgentScales(metricsData) {
    const scales = new Set();

    if (metricsData.operations) {
      metricsData.operations.forEach(op => {
        if (op.agentCount) {
          scales.add(op.agentCount);
        }
      });
    }

    return Array.from(scales).sort((a, b) => a - b);
  }
}

/**
 * Analysis Query Interface
 */
class AnalysisQuery {
  constructor(memoryStore) {
    this.memory = memoryStore;
  }

  async getTopologyPerformance(topology = null) {
    const data = await this.memory.retrieveKey('topology-performance');

    if (!data) {
      return null;
    }

    if (topology) {
      return data[topology] || null;
    }

    return data;
  }

  async getScalingMetrics(fromCount = null, toCount = null) {
    const data = await this.memory.retrieveKey('scaling-metrics');

    if (!data) {
      return null;
    }

    if (fromCount !== null && toCount !== null) {
      const key = `${fromCount}-${toCount}`;
      return data[key] || null;
    }

    return data;
  }

  async getReasoningBankMetrics() {
    return await this.memory.retrieveKey('reasoningbank-integration');
  }

  async getReliabilityResults(testType = null) {
    const data = await this.memory.retrieveKey('reliability-results');

    if (!data) {
      return null;
    }

    if (testType) {
      return data[testType] || null;
    }

    return data;
  }

  async getSummary() {
    return await this.memory.retrieveKey('summary');
  }

  async generateQuickReport() {
    const summary = await this.getSummary();

    if (!summary) {
      return 'No analysis data found in memory';
    }

    const lines = [];

    lines.push('╔════════════════════════════════════════════════════════════╗');
    lines.push('║  E2B Swarm Analysis - Quick Report (from memory)          ║');
    lines.push('╚════════════════════════════════════════════════════════════╝\n');

    lines.push(`Last Updated: ${new Date(summary.timestamp).toLocaleString()}`);
    lines.push(`Total Operations: ${summary.totalOperations}`);
    lines.push(`Success Rate: ${summary.successRate.toFixed(2)}%`);
    lines.push(`Topologies Tested: ${summary.topologiesTested.join(', ')}`);
    lines.push(`Agent Scales: ${summary.agentScalesTested.join(', ')}\n`);

    // Topology performance
    const topologyPerf = await this.getTopologyPerformance();
    if (topologyPerf) {
      lines.push('Top Performing Topologies:');

      Object.entries(topologyPerf).forEach(([topology, data]) => {
        if (data[0] && data[0].init) {
          lines.push(`  ${topology}: ${data[0].init.avg.toFixed(0)}ms init, ${data[0].deploy.avg.toFixed(0)}ms deploy`);
        }
      });
    }

    return lines.join('\n');
  }
}

/**
 * Main execution
 */
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  const memoryStore = new MemoryStoreManager();
  const query = new AnalysisQuery(memoryStore);

  switch (command) {
    case 'store':
      const metricsFile = args[1];
      if (!metricsFile) {
        console.error('Usage: memory-store.js store <metrics-file.json>');
        process.exit(1);
      }

      const metricsData = JSON.parse(await fs.readFile(metricsFile, 'utf8'));
      await memoryStore.storeMetrics(metricsData);
      break;

    case 'retrieve':
      const key = args[1];
      if (!key) {
        console.error('Usage: memory-store.js retrieve <key>');
        process.exit(1);
      }

      const data = await memoryStore.retrieveKey(key);
      console.log(JSON.stringify(data, null, 2));
      break;

    case 'list':
      const keys = await memoryStore.listStoredKeys();
      console.log('Stored keys:');
      keys.forEach(k => console.log(`  - ${memoryStore.namespace}/${k}`));
      break;

    case 'report':
      const report = await query.generateQuickReport();
      console.log(report);
      break;

    default:
      console.log('E2B Swarm Memory Store\n');
      console.log('Usage:');
      console.log('  memory-store.js store <metrics-file.json>  - Store metrics in memory');
      console.log('  memory-store.js retrieve <key>             - Retrieve stored data');
      console.log('  memory-store.js list                       - List all stored keys');
      console.log('  memory-store.js report                     - Generate quick report');
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  MemoryStoreManager,
  AnalysisQuery
};
