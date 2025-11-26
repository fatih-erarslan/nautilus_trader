/**
 * Tool Tests
 * Verify all 99+ tools are accessible and return expected formats
 */

const { describe, it } = require('mocha');
const { expect } = require('chai');
const { ToolRegistry } = require('../src/discovery/registry');
const path = require('path');

describe('Tool Registry', function() {
  let registry;

  before(async function() {
    registry = new ToolRegistry({
      toolsDir: path.join(__dirname, '../tools')
    });
    await registry.loadTools();
  });

  describe('Tool Loading', function() {
    it('should load all tool schemas', function() {
      expect(registry.tools.size).to.be.greaterThan(80); // At least 80+ tools
    });

    it('should have ETag for each tool', function() {
      for (const [name] of registry.tools) {
        const etag = registry.getToolETag(name);
        expect(etag).to.be.a('string');
        expect(etag).to.have.lengthOf(64); // SHA-256 hex
      }
    });
  });

  describe('Tool Categories', function() {
    const expectedCategories = [
      'Trading',
      'Neural Networks',
      'News Trading',
      'Portfolio & Risk',
      'Sports Betting',
      'Prediction Markets',
      'Syndicates',
      'E2B Cloud'
    ];

    expectedCategories.forEach(category => {
      it(`should have ${category} tools`, function() {
        const tools = registry.getToolsByCategory(category);
        expect(tools.length).to.be.greaterThan(0);
      });
    });
  });

  describe('Core Trading Tools', function() {
    const tradingTools = [
      'list_strategies',
      'execute_trade',
      'run_backtest',
      'optimize_strategy',
      'quick_analysis'
    ];

    tradingTools.forEach(toolName => {
      it(`should have ${toolName} tool`, function() {
        expect(registry.hasTool(toolName)).to.be.true;
        const schema = registry.getToolSchema(toolName);
        expect(schema).to.exist;
        expect(schema).to.have.property('$schema');
        expect(schema).to.have.property('title', toolName);
      });
    });
  });

  describe('Neural Network Tools', function() {
    const neuralTools = [
      'neural_train',
      'neural_forecast',
      'neural_evaluate',
      'neural_optimize',
      'neural_model_status'
    ];

    neuralTools.forEach(toolName => {
      it(`should have ${toolName} tool`, function() {
        expect(registry.hasTool(toolName)).to.be.true;
      });
    });
  });

  describe('Sports Betting Tools', function() {
    const sportsTools = [
      'get_sports_events',
      'get_sports_odds',
      'find_sports_arbitrage',
      'calculate_kelly_criterion',
      'execute_sports_bet'
    ];

    sportsTools.forEach(toolName => {
      it(`should have ${toolName} tool`, function() {
        expect(registry.hasTool(toolName)).to.be.true;
      });
    });
  });

  describe('E2B Cloud Tools', function() {
    const e2bTools = [
      'create_e2b_sandbox',
      'run_e2b_agent',
      'execute_e2b_process',
      'list_e2b_sandboxes',
      'terminate_e2b_sandbox'
    ];

    e2bTools.forEach(toolName => {
      it(`should have ${toolName} tool`, function() {
        expect(registry.hasTool(toolName)).to.be.true;
      });
    });
  });

  describe('Syndicate Tools', function() {
    const syndicateTools = [
      'create_syndicate',
      'add_syndicate_member',
      'get_syndicate_status',
      'allocate_syndicate_funds',
      'distribute_syndicate_profits'
    ];

    syndicateTools.forEach(toolName => {
      it(`should have ${toolName} tool`, function() {
        expect(registry.hasTool(toolName)).to.be.true;
      });
    });
  });

  describe('Tool Search', function() {
    it('should search by keyword', function() {
      const results = registry.searchTools('neural');
      expect(results.length).to.be.greaterThan(0);
      results.forEach(name => {
        expect(name.toLowerCase()).to.include('neural');
      });
    });

    it('should search by category', function() {
      const results = registry.searchTools('trading');
      expect(results.length).to.be.greaterThan(0);
    });

    it('should return empty for unknown keyword', function() {
      const results = registry.searchTools('xyznonexistent');
      expect(results).to.be.an('array');
      expect(results.length).to.equal(0);
    });
  });

  describe('Schema Validation', function() {
    it('should have valid JSON Schema for all tools', function() {
      const tools = registry.listTools();

      tools.forEach(tool => {
        const schema = registry.getToolSchema(tool.name);
        expect(schema).to.have.property('$schema');
        expect(schema.$schema).to.include('json-schema.org');
        expect(schema).to.have.property('title');
        expect(schema).to.have.property('description');
      });
    });

    it('should have metadata for all tools', function() {
      const tools = registry.listTools();

      tools.forEach(tool => {
        const metadata = registry.getToolMetadata(tool.name);
        expect(metadata).to.exist;
        if (metadata.cost) {
          expect(['low', 'medium', 'high']).to.include(metadata.cost);
        }
      });
    });
  });
});
