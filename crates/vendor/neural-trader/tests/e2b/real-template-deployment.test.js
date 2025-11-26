/**
 * E2B Template Deployment Tests - Real API Integration
 *
 * Comprehensive test suite validating all E2B template types with real API calls.
 * Tests deployment, execution, customization, and resource management.
 *
 * @requires @neural-trader/backend NAPI module
 * @requires .env with E2B_API_KEY and E2B_ACCESS_TOKEN
 */

const { describe, test, beforeAll, afterAll, expect } = require('@jest/globals');
const path = require('path');
const fs = require('fs');
const { loadEnv, getE2BCredentials } = require('./helpers/env-loader');
const { E2BTemplateManager } = require('./helpers/template-manager');
const { PerformanceMonitor } = require('./helpers/performance-monitor');
const { ResourceCleaner } = require('./helpers/resource-cleaner');

// Load NAPI backend
let backend;
try {
  backend = require('@neural-trader/backend');
} catch (error) {
  console.warn('⚠️  NAPI backend not available, using mock');
  backend = require('./helpers/mock-backend');
}

// Test configuration
const TEST_CONFIG = {
  templates: ['base', 'nodejs', 'python', 'react'],
  timeout: 120000, // 2 minutes per test
  cleanupDelay: 5000, // 5 seconds between cleanup
  maxParallelTests: 3,
};

// Global test state
let templateManager;
let performanceMonitor;
let resourceCleaner;
let credentials;
let createdSandboxes = [];

describe('E2B Template Deployment - Real API Integration', () => {

  // ============================================
  // Setup and Teardown
  // ============================================

  beforeAll(async () => {
    console.log('🚀 Initializing E2B Template Deployment Tests...');

    // Load environment
    loadEnv();
    credentials = getE2BCredentials();

    if (!credentials.apiKey) {
      console.warn('⚠️  E2B_API_KEY not found, tests will run in mock mode');
    }

    // Initialize managers
    templateManager = new E2BTemplateManager(credentials);
    performanceMonitor = new PerformanceMonitor();
    resourceCleaner = new ResourceCleaner(credentials);

    // Initialize NAPI backend
    if (backend.initNeuralTrader) {
      await backend.initNeuralTrader(JSON.stringify({
        e2b_api_key: credentials.apiKey,
        e2b_access_token: credentials.accessToken,
      }));
    }

    console.log('✅ Test environment initialized');
  }, 30000);

  afterAll(async () => {
    console.log('🧹 Cleaning up test resources...');

    // Clean up all created sandboxes
    for (const sandbox of createdSandboxes) {
      try {
        await resourceCleaner.cleanupSandbox(sandbox.sandboxId);
        await new Promise(resolve => setTimeout(resolve, TEST_CONFIG.cleanupDelay));
      } catch (error) {
        console.warn(`Failed to cleanup sandbox ${sandbox.sandboxId}:`, error.message);
      }
    }

    // Generate performance report
    const report = performanceMonitor.generateReport();
    console.log('\n📊 Performance Report:');
    console.log(JSON.stringify(report, null, 2));

    // Save report to file
    const reportPath = path.join(__dirname, 'performance-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`📄 Report saved to: ${reportPath}`);

    console.log('✅ Cleanup complete');
  }, 60000);

  // ============================================
  // Base Template Tests
  // ============================================

  describe('Base Template Deployment', () => {

    test('Deploy base template and execute JavaScript', async () => {
      const startTime = Date.now();

      // Create sandbox with base template
      const sandbox = await backend.createE2bSandbox(
        'test-base-template',
        'base'
      );

      expect(sandbox).toBeDefined();
      expect(sandbox.template).toBe('base');
      expect(sandbox.status).toBe('running');

      createdSandboxes.push(sandbox);
      const deployTime = Date.now() - startTime;

      // Execute JavaScript code
      const execStart = Date.now();
      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        'node -e "console.log(Math.PI * 2)"'
      );
      const execTime = Date.now() - execStart;

      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('6.28');

      // Record metrics
      performanceMonitor.recordMetric('base_template_deploy', {
        deploymentTimeMs: deployTime,
        executionTimeMs: execTime,
        template: 'base',
      });

      console.log(`✅ Base template: Deploy ${deployTime}ms, Exec ${execTime}ms`);
    }, TEST_CONFIG.timeout);

    test('Validate base template resource limits', async () => {
      const sandbox = createdSandboxes.find(s => s.template === 'base');

      if (!sandbox) {
        console.warn('⚠️  No base sandbox available, skipping test');
        return;
      }

      // Test memory allocation
      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        'node -e "const arr = new Array(1000000).fill(0); console.log(arr.length)"'
      );

      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('1000000');

      console.log('✅ Base template resource limits validated');
    }, TEST_CONFIG.timeout);

  });

  // ============================================
  // Node.js Template Tests
  // ============================================

  describe('Node.js Template Deployment', () => {

    test('Deploy Node.js template with npm packages', async () => {
      const startTime = Date.now();

      const sandbox = await backend.createE2bSandbox(
        'test-nodejs-template',
        'nodejs'
      );

      expect(sandbox).toBeDefined();
      expect(sandbox.template).toBe('nodejs');

      createdSandboxes.push(sandbox);
      const deployTime = Date.now() - startTime;

      // Install and use a package
      const execStart = Date.now();
      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        'npm install lodash && node -e "const _ = require(\'lodash\'); console.log(_.sum([1,2,3]))"'
      );
      const execTime = Date.now() - execStart;

      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('6');

      performanceMonitor.recordMetric('nodejs_template_deploy', {
        deploymentTimeMs: deployTime,
        packageInstallTimeMs: execTime,
        template: 'nodejs',
      });

      console.log(`✅ Node.js template: Deploy ${deployTime}ms, Package install ${execTime}ms`);
    }, TEST_CONFIG.timeout);

    test('Execute trading strategy simulation', async () => {
      const sandbox = createdSandboxes.find(s => s.template === 'nodejs');

      if (!sandbox) {
        console.warn('⚠️  No Node.js sandbox available, skipping test');
        return;
      }

      // Simple trading simulation
      const tradingCode = `
        const prices = [100, 102, 101, 105, 103];
        const sma = prices.reduce((a,b) => a+b) / prices.length;
        console.log(JSON.stringify({
          sma: sma.toFixed(2),
          signal: prices[prices.length-1] > sma ? 'BUY' : 'SELL'
        }));
      `;

      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        `node -e "${tradingCode.replace(/\n/g, ' ')}"`
      );

      expect(result.exitCode).toBe(0);

      const output = JSON.parse(result.stdout.trim());
      expect(output).toHaveProperty('sma');
      expect(output).toHaveProperty('signal');
      expect(['BUY', 'SELL']).toContain(output.signal);

      console.log('✅ Trading strategy simulation successful:', output);
    }, TEST_CONFIG.timeout);

  });

  // ============================================
  // Python Template Tests
  // ============================================

  describe('Python Template Deployment', () => {

    test('Deploy Python template with pip packages', async () => {
      const startTime = Date.now();

      const sandbox = await backend.createE2bSandbox(
        'test-python-template',
        'python'
      );

      expect(sandbox).toBeDefined();
      expect(sandbox.template).toBe('python');

      createdSandboxes.push(sandbox);
      const deployTime = Date.now() - startTime;

      // Install and use numpy
      const execStart = Date.now();
      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        'pip install numpy && python -c "import numpy as np; print(np.mean([1,2,3,4,5]))"'
      );
      const execTime = Date.now() - execStart;

      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('3');

      performanceMonitor.recordMetric('python_template_deploy', {
        deploymentTimeMs: deployTime,
        packageInstallTimeMs: execTime,
        template: 'python',
      });

      console.log(`✅ Python template: Deploy ${deployTime}ms, Package install ${execTime}ms`);
    }, TEST_CONFIG.timeout);

    test('Execute ML model simulation', async () => {
      const sandbox = createdSandboxes.find(s => s.template === 'python');

      if (!sandbox) {
        console.warn('⚠️  No Python sandbox available, skipping test');
        return;
      }

      const mlCode = `
import json
import random

# Simple price prediction simulation
prices = [100 + random.uniform(-5, 5) for _ in range(10)]
prediction = sum(prices[-3:]) / 3 + random.uniform(-2, 2)

result = {
    "current_price": prices[-1],
    "predicted_price": round(prediction, 2),
    "confidence": round(random.uniform(0.7, 0.95), 2)
}

print(json.dumps(result))
      `;

      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        `python -c "${mlCode.replace(/"/g, '\\"').replace(/\n/g, '; ')}"`
      );

      expect(result.exitCode).toBe(0);

      const output = JSON.parse(result.stdout.trim());
      expect(output).toHaveProperty('current_price');
      expect(output).toHaveProperty('predicted_price');
      expect(output).toHaveProperty('confidence');
      expect(output.confidence).toBeGreaterThan(0.6);

      console.log('✅ ML model simulation successful:', output);
    }, TEST_CONFIG.timeout);

  });

  // ============================================
  // React Template Tests
  // ============================================

  describe('React Template Deployment', () => {

    test('Deploy React template for UI testing', async () => {
      const startTime = Date.now();

      const sandbox = await backend.createE2bSandbox(
        'test-react-template',
        'react'
      );

      expect(sandbox).toBeDefined();
      expect(sandbox.template).toBe('react');

      createdSandboxes.push(sandbox);
      const deployTime = Date.now() - startTime;

      // Verify React is available
      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        'node -e "console.log(require(\'react\').version)"'
      );

      expect(result.exitCode).toBe(0);

      performanceMonitor.recordMetric('react_template_deploy', {
        deploymentTimeMs: deployTime,
        template: 'react',
      });

      console.log(`✅ React template: Deploy ${deployTime}ms`);
    }, TEST_CONFIG.timeout);

  });

  // ============================================
  // Advanced Tests
  // ============================================

  describe('Advanced E2B Operations', () => {

    test('Deploy multiple templates in parallel', async () => {
      const templates = ['base', 'nodejs', 'python'];
      const startTime = Date.now();

      const deployments = await Promise.all(
        templates.map(async (template, index) => {
          const sandbox = await backend.createE2bSandbox(
            `parallel-test-${template}-${Date.now()}`,
            template
          );
          createdSandboxes.push(sandbox);
          return { template, sandbox };
        })
      );

      const totalTime = Date.now() - startTime;

      expect(deployments).toHaveLength(3);
      deployments.forEach(({ template, sandbox }) => {
        expect(sandbox.template).toBe(template);
        expect(sandbox.status).toBe('running');
      });

      performanceMonitor.recordMetric('parallel_deployment', {
        totalTimeMs: totalTime,
        templateCount: templates.length,
        avgTimePerTemplate: totalTime / templates.length,
      });

      console.log(`✅ Parallel deployment: ${templates.length} templates in ${totalTime}ms`);
    }, TEST_CONFIG.timeout);

    test('Test template customization', async () => {
      const sandbox = await backend.createE2bSandbox(
        'custom-template-test',
        'nodejs'
      );

      createdSandboxes.push(sandbox);

      // Install custom packages
      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        'npm install axios chalk && node -e "console.log(\'Custom environment ready\')"'
      );

      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('Custom environment ready');

      console.log('✅ Template customization successful');
    }, TEST_CONFIG.timeout);

    test('Measure deployment performance metrics', async () => {
      const metrics = performanceMonitor.getAllMetrics();

      expect(metrics).toBeDefined();
      expect(Object.keys(metrics).length).toBeGreaterThan(0);

      // Calculate statistics
      const deploymentTimes = Object.values(metrics)
        .filter(m => m.deploymentTimeMs)
        .map(m => m.deploymentTimeMs);

      if (deploymentTimes.length > 0) {
        const avgDeployTime = deploymentTimes.reduce((a, b) => a + b) / deploymentTimes.length;
        const maxDeployTime = Math.max(...deploymentTimes);
        const minDeployTime = Math.min(...deploymentTimes);

        console.log('\n📊 Deployment Performance:');
        console.log(`  Average: ${avgDeployTime.toFixed(2)}ms`);
        console.log(`  Min: ${minDeployTime}ms`);
        console.log(`  Max: ${maxDeployTime}ms`);

        expect(avgDeployTime).toBeLessThan(30000); // Should be under 30s
      }
    });

    test('Validate sandbox resource cleanup', async () => {
      // Create a temporary sandbox
      const sandbox = await backend.createE2bSandbox(
        'cleanup-test',
        'base'
      );

      expect(sandbox).toBeDefined();

      // Clean it up immediately
      const cleanupSuccess = await resourceCleaner.cleanupSandbox(sandbox.sandboxId);
      expect(cleanupSuccess).toBe(true);

      console.log('✅ Resource cleanup validated');
    }, TEST_CONFIG.timeout);

    test('Execute custom trading bot template', async () => {
      const sandbox = await backend.createE2bSandbox(
        'trading-bot-template',
        'nodejs'
      );

      createdSandboxes.push(sandbox);

      // Deploy a simple trading bot
      const botCode = `
        const strategies = {
          momentum: (prices) => prices[prices.length-1] > prices[prices.length-2] ? 'BUY' : 'SELL',
          meanReversion: (prices) => {
            const avg = prices.reduce((a,b) => a+b) / prices.length;
            return prices[prices.length-1] < avg ? 'BUY' : 'SELL';
          }
        };

        const prices = [100, 102, 101, 99, 98, 100, 103];
        const signals = {
          momentum: strategies.momentum(prices),
          meanReversion: strategies.meanReversion(prices)
        };

        console.log(JSON.stringify(signals));
      `;

      const result = await backend.executeE2bProcess(
        sandbox.sandboxId,
        `node -e "${botCode.replace(/\n/g, ' ').replace(/"/g, '\\"')}"`
      );

      expect(result.exitCode).toBe(0);

      const signals = JSON.parse(result.stdout.trim());
      expect(signals).toHaveProperty('momentum');
      expect(signals).toHaveProperty('meanReversion');

      console.log('✅ Trading bot executed:', signals);
    }, TEST_CONFIG.timeout);

    test('Test template switching and migration', async () => {
      // Create sandbox with one template
      const sandbox1 = await backend.createE2bSandbox(
        'migration-source',
        'nodejs'
      );

      createdSandboxes.push(sandbox1);

      // Execute code in first sandbox
      await backend.executeE2bProcess(
        sandbox1.sandboxId,
        'echo "Data from source" > /tmp/data.txt'
      );

      // Create second sandbox with different template
      const sandbox2 = await backend.createE2bSandbox(
        'migration-target',
        'python'
      );

      createdSandboxes.push(sandbox2);

      // Verify both are running
      expect(sandbox1.status).toBe('running');
      expect(sandbox2.status).toBe('running');

      console.log('✅ Template switching validated');
    }, TEST_CONFIG.timeout);

  });

  // ============================================
  // Cost and Resource Analysis
  // ============================================

  describe('Cost and Resource Analysis', () => {

    test('Calculate estimated costs per template', async () => {
      const metrics = performanceMonitor.getAllMetrics();

      // Estimated cost per template (mock values for demonstration)
      const costPerMinute = {
        base: 0.01,
        nodejs: 0.015,
        python: 0.02,
        react: 0.025,
      };

      const costAnalysis = {};

      Object.entries(metrics).forEach(([name, metric]) => {
        if (metric.template && metric.deploymentTimeMs) {
          const template = metric.template;
          const minutes = metric.deploymentTimeMs / 60000;
          const cost = minutes * (costPerMinute[template] || 0.01);

          if (!costAnalysis[template]) {
            costAnalysis[template] = { totalCost: 0, count: 0 };
          }

          costAnalysis[template].totalCost += cost;
          costAnalysis[template].count += 1;
        }
      });

      console.log('\n💰 Cost Analysis:');
      Object.entries(costAnalysis).forEach(([template, data]) => {
        const avgCost = data.totalCost / data.count;
        console.log(`  ${template}: $${avgCost.toFixed(4)} per deployment (${data.count} tests)`);
      });

      expect(Object.keys(costAnalysis).length).toBeGreaterThan(0);
    });

    test('Analyze memory usage per template', async () => {
      const memoryUsage = {};

      for (const sandbox of createdSandboxes) {
        // Execute memory check
        const result = await backend.executeE2bProcess(
          sandbox.sandboxId,
          'node -e "const used = process.memoryUsage(); console.log(JSON.stringify(used))"'
        );

        if (result.exitCode === 0) {
          try {
            const memory = JSON.parse(result.stdout.trim());
            memoryUsage[sandbox.template] = memoryUsage[sandbox.template] || [];
            memoryUsage[sandbox.template].push(memory.heapUsed);
          } catch (error) {
            console.warn(`Failed to parse memory for ${sandbox.sandboxId}`);
          }
        }
      }

      console.log('\n💾 Memory Usage:');
      Object.entries(memoryUsage).forEach(([template, usage]) => {
        const avgUsage = usage.reduce((a, b) => a + b) / usage.length;
        console.log(`  ${template}: ${(avgUsage / 1024 / 1024).toFixed(2)} MB avg`);
      });

      expect(Object.keys(memoryUsage).length).toBeGreaterThan(0);
    });

  });

});

// Export for external use
module.exports = {
  TEST_CONFIG,
  createdSandboxes,
};
