/**
 * Tests for model selector
 */

import { ModelSelector } from '../src/model-selector';

describe('ModelSelector', () => {
  let selector: ModelSelector;

  beforeEach(() => {
    selector = new ModelSelector();
  });

  describe('selectModel', () => {
    it('should select appropriate model for simple task', () => {
      const model = selector.selectModel({
        complexity: 'simple',
      });

      expect(model).toBeDefined();
      expect(model.bestFor).toContain('simple');
    });

    it('should select model within cost constraints', () => {
      const model = selector.selectModel({
        complexity: 'moderate',
        maxCost: 0.000001, // Very low cost per token
      });

      expect(model).toBeDefined();
      const avgCost = (model.cost.prompt + model.cost.completion) / 2;
      expect(avgCost).toBeLessThanOrEqual(1.0);
    });

    it('should select model with code generation capability', () => {
      const model = selector.selectModel({
        complexity: 'complex',
        requiresCodeGeneration: true,
      });

      expect(model.strengths).toContain('code');
    });

    it('should throw error when no models match', () => {
      expect(() =>
        selector.selectModel({
          complexity: 'simple',
          maxLatency: 1, // Impossibly low latency
        })
      ).toThrow('No models match');
    });
  });

  describe('compareModels', () => {
    it('should compare models and return sorted results', () => {
      const comparison = selector.compareModels({
        complexity: 'moderate',
      });

      expect(comparison).toBeInstanceOf(Array);
      expect(comparison.length).toBeGreaterThan(0);
      expect(comparison[0]).toHaveProperty('model');
      expect(comparison[0]).toHaveProperty('score');
      expect(comparison[0]).toHaveProperty('costEfficiency');

      // Verify sorting (higher score first)
      for (let i = 0; i < comparison.length - 1; i++) {
        expect(comparison[i].score).toBeGreaterThanOrEqual(comparison[i + 1].score);
      }
    });
  });

  describe('estimateModelCost', () => {
    it('should calculate cost for specific model', () => {
      const cost = selector.estimateModelCost(
        'openai/gpt-3.5-turbo',
        1000,
        500
      );

      expect(cost).toHaveProperty('promptCost');
      expect(cost).toHaveProperty('completionCost');
      expect(cost).toHaveProperty('totalCost');
      expect(cost.totalCost).toBe(cost.promptCost + cost.completionCost);
    });
  });
});
