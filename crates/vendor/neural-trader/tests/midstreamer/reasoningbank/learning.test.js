/**
 * ReasoningBank Learning Tests
 * Tests adaptive learning, memory distillation, and pattern recognition
 */

const { performance } = require('perf_hooks');

describe('ReasoningBank Learning', () => {
  let reasoningBank;

  beforeEach(() => {
    // Mock ReasoningBank implementation
    reasoningBank = {
      experiences: [],
      memory: new Map(),
      thresholds: {
        success: 0.7,
        failure: 0.3,
        confidence: 0.6
      },

      recordExperience: function(experience) {
        const record = {
          id: this.experiences.length,
          timestamp: Date.now(),
          trajectory: experience.trajectory,
          context: experience.context,
          outcome: null,
          verdict: null,
          learned: false,
          ...experience
        };

        this.experiences.push(record);
        return record.id;
      },

      updateOutcome: function(experienceId, outcome) {
        if (experienceId >= this.experiences.length) {
          throw new Error('Experience not found');
        }

        this.experiences[experienceId].outcome = outcome;
        this.experiences[experienceId].outcomeTimestamp = Date.now();

        return this.experiences[experienceId];
      },

      judgeVerdict: function(experienceId) {
        const exp = this.experiences[experienceId];
        if (!exp.outcome) {
          throw new Error('Outcome not set');
        }

        const { outcome } = exp;
        let verdict = 'NEUTRAL';

        if (outcome.success >= this.thresholds.success) {
          verdict = 'SUCCESS';
        } else if (outcome.success <= this.thresholds.failure) {
          verdict = 'FAILURE';
        }

        exp.verdict = verdict;
        exp.verdictTimestamp = Date.now();
        exp.confidence = outcome.confidence || 0.5;

        return verdict;
      },

      distillMemory: function(experienceIds) {
        const patterns = new Map();

        for (const id of experienceIds) {
          const exp = this.experiences[id];
          if (!exp.verdict) continue;

          const patternKey = JSON.stringify(exp.context);

          if (!patterns.has(patternKey)) {
            patterns.set(patternKey, {
              context: exp.context,
              successes: 0,
              failures: 0,
              neutrals: 0,
              trajectories: []
            });
          }

          const pattern = patterns.get(patternKey);
          pattern.trajectories.push(exp.trajectory);

          if (exp.verdict === 'SUCCESS') pattern.successes++;
          else if (exp.verdict === 'FAILURE') pattern.failures++;
          else pattern.neutrals++;
        }

        // Store distilled patterns
        for (const [key, pattern] of patterns) {
          this.memory.set(key, {
            ...pattern,
            total: pattern.successes + pattern.failures + pattern.neutrals,
            successRate: pattern.successes / (pattern.successes + pattern.failures + pattern.neutrals),
            distilledAt: Date.now()
          });
        }

        return Array.from(this.memory.values());
      },

      adaptThresholds: function(performanceMetrics) {
        const { successRate, avgConfidence, volatility } = performanceMetrics;

        // Adaptive threshold adjustment
        if (successRate > 0.8 && avgConfidence > 0.7) {
          // System is performing well, raise thresholds
          this.thresholds.success = Math.min(0.85, this.thresholds.success + 0.05);
          this.thresholds.failure = Math.max(0.15, this.thresholds.failure - 0.05);
        } else if (successRate < 0.5 || avgConfidence < 0.5) {
          // System struggling, lower thresholds to be more conservative
          this.thresholds.success = Math.max(0.6, this.thresholds.success - 0.05);
          this.thresholds.failure = Math.min(0.4, this.thresholds.failure + 0.05);
        }

        // Adjust confidence threshold based on volatility
        if (volatility > 0.3) {
          this.thresholds.confidence = Math.min(0.8, this.thresholds.confidence + 0.1);
        } else if (volatility < 0.1) {
          this.thresholds.confidence = Math.max(0.5, this.thresholds.confidence - 0.05);
        }

        return this.thresholds;
      },

      query: function(context) {
        const patternKey = JSON.stringify(context);
        return this.memory.get(patternKey) || null;
      }
    };
  });

  describe('Experience Recording', () => {
    it('should record new experience with trajectory', () => {
      const experience = {
        trajectory: ['ANALYZE', 'BUY', 'MONITOR', 'SELL'],
        context: { market: 'BULLISH', volatility: 'LOW' }
      };

      const id = reasoningBank.recordExperience(experience);

      expect(id).toBe(0);
      expect(reasoningBank.experiences[0].trajectory).toEqual(experience.trajectory);
      expect(reasoningBank.experiences[0].context).toEqual(experience.context);
      expect(reasoningBank.experiences[0].outcome).toBeNull();
    });

    it('should assign sequential IDs to experiences', () => {
      const exp1 = reasoningBank.recordExperience({ trajectory: ['A'], context: {} });
      const exp2 = reasoningBank.recordExperience({ trajectory: ['B'], context: {} });
      const exp3 = reasoningBank.recordExperience({ trajectory: ['C'], context: {} });

      expect(exp1).toBe(0);
      expect(exp2).toBe(1);
      expect(exp3).toBe(2);
    });

    it('should preserve complex context data', () => {
      const experience = {
        trajectory: ['TRADE'],
        context: {
          market: { trend: 'UP', strength: 0.8 },
          indicators: { rsi: 45, macd: 'BULLISH' },
          risk: { stopLoss: 0.02, takeProfit: 0.06 }
        }
      };

      const id = reasoningBank.recordExperience(experience);

      expect(reasoningBank.experiences[id].context).toEqual(experience.context);
      expect(reasoningBank.experiences[id].context.indicators.macd).toBe('BULLISH');
    });
  });

  describe('Outcome Updates', () => {
    it('should update experience outcome', () => {
      const id = reasoningBank.recordExperience({
        trajectory: ['BUY', 'SELL'],
        context: { market: 'NEUTRAL' }
      });

      const outcome = {
        success: 0.85,
        profit: 1500,
        confidence: 0.9
      };

      const updated = reasoningBank.updateOutcome(id, outcome);

      expect(updated.outcome).toEqual(outcome);
      expect(updated.outcomeTimestamp).toBeDefined();
    });

    it('should throw error for invalid experience ID', () => {
      expect(() => {
        reasoningBank.updateOutcome(999, { success: 0.5 });
      }).toThrow('Experience not found');
    });

    it('should allow multiple outcome updates', () => {
      const id = reasoningBank.recordExperience({
        trajectory: ['TRADE'],
        context: {}
      });

      reasoningBank.updateOutcome(id, { success: 0.5 });
      reasoningBank.updateOutcome(id, { success: 0.8 });

      expect(reasoningBank.experiences[id].outcome.success).toBe(0.8);
    });
  });

  describe('Verdict Judgment', () => {
    it('should judge SUCCESS for high success rate', () => {
      const id = reasoningBank.recordExperience({
        trajectory: ['WINNING_TRADE'],
        context: {}
      });

      reasoningBank.updateOutcome(id, { success: 0.9, confidence: 0.8 });
      const verdict = reasoningBank.judgeVerdict(id);

      expect(verdict).toBe('SUCCESS');
      expect(reasoningBank.experiences[id].verdict).toBe('SUCCESS');
    });

    it('should judge FAILURE for low success rate', () => {
      const id = reasoningBank.recordExperience({
        trajectory: ['LOSING_TRADE'],
        context: {}
      });

      reasoningBank.updateOutcome(id, { success: 0.1, confidence: 0.6 });
      const verdict = reasoningBank.judgeVerdict(id);

      expect(verdict).toBe('FAILURE');
    });

    it('should judge NEUTRAL for moderate success', () => {
      const id = reasoningBank.recordExperience({
        trajectory: ['BREAK_EVEN'],
        context: {}
      });

      reasoningBank.updateOutcome(id, { success: 0.5, confidence: 0.5 });
      const verdict = reasoningBank.judgeVerdict(id);

      expect(verdict).toBe('NEUTRAL');
    });

    it('should throw error if outcome not set', () => {
      const id = reasoningBank.recordExperience({
        trajectory: ['TRADE'],
        context: {}
      });

      expect(() => {
        reasoningBank.judgeVerdict(id);
      }).toThrow('Outcome not set');
    });

    it('should record confidence with verdict', () => {
      const id = reasoningBank.recordExperience({
        trajectory: ['TRADE'],
        context: {}
      });

      reasoningBank.updateOutcome(id, { success: 0.95, confidence: 0.88 });
      reasoningBank.judgeVerdict(id);

      expect(reasoningBank.experiences[id].confidence).toBe(0.88);
    });
  });

  describe('Memory Distillation', () => {
    it('should distill patterns from multiple experiences', () => {
      // Record similar experiences
      const context = { market: 'BULLISH', volatility: 'LOW' };

      for (let i = 0; i < 5; i++) {
        const id = reasoningBank.recordExperience({
          trajectory: ['BUY', 'HOLD', 'SELL'],
          context
        });
        reasoningBank.updateOutcome(id, { success: 0.8 });
        reasoningBank.judgeVerdict(id);
      }

      const patterns = reasoningBank.distillMemory([0, 1, 2, 3, 4]);

      expect(patterns.length).toBe(1);
      expect(patterns[0].total).toBe(5);
      expect(patterns[0].successes).toBe(5);
    });

    it('should calculate success rate from distilled patterns', () => {
      const context = { market: 'VOLATILE' };

      // 7 successes, 3 failures
      for (let i = 0; i < 10; i++) {
        const id = reasoningBank.recordExperience({ trajectory: ['TRADE'], context });
        const success = i < 7 ? 0.9 : 0.1;
        reasoningBank.updateOutcome(id, { success });
        reasoningBank.judgeVerdict(id);
      }

      const patterns = reasoningBank.distillMemory([...Array(10).keys()]);

      expect(patterns[0].successRate).toBeCloseTo(0.7, 1);
      expect(patterns[0].successes).toBe(7);
      expect(patterns[0].failures).toBe(3);
    });

    it('should group experiences by context', () => {
      // Different contexts
      const contexts = [
        { market: 'BULLISH' },
        { market: 'BEARISH' },
        { market: 'BULLISH' }
      ];

      const ids = [];
      for (const context of contexts) {
        const id = reasoningBank.recordExperience({ trajectory: ['TRADE'], context });
        reasoningBank.updateOutcome(id, { success: 0.8 });
        reasoningBank.judgeVerdict(id);
        ids.push(id);
      }

      const patterns = reasoningBank.distillMemory(ids);

      expect(patterns.length).toBe(2); // Two unique contexts
    });

    it('should preserve trajectory information in patterns', () => {
      const context = { strategy: 'SCALPING' };
      const trajectories = [
        ['QUICK_BUY', 'QUICK_SELL'],
        ['FAST_ENTRY', 'FAST_EXIT'],
        ['RAPID_TRADE']
      ];

      const ids = [];
      for (const trajectory of trajectories) {
        const id = reasoningBank.recordExperience({ trajectory, context });
        reasoningBank.updateOutcome(id, { success: 0.8 });
        reasoningBank.judgeVerdict(id);
        ids.push(id);
      }

      const patterns = reasoningBank.distillMemory(ids);

      expect(patterns[0].trajectories.length).toBe(3);
      expect(patterns[0].trajectories).toContainEqual(trajectories[0]);
    });

    it('should allow querying distilled patterns', () => {
      const context = { market: 'BULLISH', indicator: 'RSI_OVERSOLD' };

      const id = reasoningBank.recordExperience({ trajectory: ['BUY'], context });
      reasoningBank.updateOutcome(id, { success: 0.9 });
      reasoningBank.judgeVerdict(id);
      reasoningBank.distillMemory([id]);

      const pattern = reasoningBank.query(context);

      expect(pattern).toBeDefined();
      expect(pattern.successRate).toBeGreaterThan(0.8);
    });
  });

  describe('Adaptive Threshold Changes', () => {
    it('should raise thresholds when performing well', () => {
      const initialThresholds = { ...reasoningBank.thresholds };

      reasoningBank.adaptThresholds({
        successRate: 0.85,
        avgConfidence: 0.75,
        volatility: 0.15
      });

      expect(reasoningBank.thresholds.success).toBeGreaterThan(initialThresholds.success);
      expect(reasoningBank.thresholds.failure).toBeLessThan(initialThresholds.failure);
    });

    it('should lower thresholds when struggling', () => {
      const initialThresholds = { ...reasoningBank.thresholds };

      reasoningBank.adaptThresholds({
        successRate: 0.4,
        avgConfidence: 0.45,
        volatility: 0.2
      });

      expect(reasoningBank.thresholds.success).toBeLessThan(initialThresholds.success);
      expect(reasoningBank.thresholds.failure).toBeGreaterThan(initialThresholds.failure);
    });

    it('should adjust confidence threshold based on volatility', () => {
      const initialConfidence = reasoningBank.thresholds.confidence;

      reasoningBank.adaptThresholds({
        successRate: 0.6,
        avgConfidence: 0.6,
        volatility: 0.4 // High volatility
      });

      expect(reasoningBank.thresholds.confidence).toBeGreaterThan(initialConfidence);
    });

    it('should not exceed threshold bounds', () => {
      // Try to push thresholds to extremes
      for (let i = 0; i < 10; i++) {
        reasoningBank.adaptThresholds({
          successRate: 0.95,
          avgConfidence: 0.9,
          volatility: 0.05
        });
      }

      expect(reasoningBank.thresholds.success).toBeLessThanOrEqual(0.85);
      expect(reasoningBank.thresholds.failure).toBeGreaterThanOrEqual(0.15);
    });

    it('should adapt over time with mixed performance', () => {
      const thresholdHistory = [{ ...reasoningBank.thresholds }];

      // Simulate varying performance
      const performances = [
        { successRate: 0.8, avgConfidence: 0.7, volatility: 0.1 },
        { successRate: 0.5, avgConfidence: 0.5, volatility: 0.3 },
        { successRate: 0.7, avgConfidence: 0.65, volatility: 0.15 },
        { successRate: 0.85, avgConfidence: 0.8, volatility: 0.1 }
      ];

      for (const perf of performances) {
        reasoningBank.adaptThresholds(perf);
        thresholdHistory.push({ ...reasoningBank.thresholds });
      }

      expect(thresholdHistory.length).toBe(5);
      // Thresholds should have changed
      expect(thresholdHistory[0]).not.toEqual(thresholdHistory[4]);
    });
  });

  describe('Integration: Complete Learning Cycle', () => {
    it('should complete full learning cycle', () => {
      // 1. Record experiences
      const context = { market: 'TRENDING', timeframe: '15m' };
      const ids = [];

      for (let i = 0; i < 10; i++) {
        const id = reasoningBank.recordExperience({
          trajectory: ['ENTER', 'MANAGE', 'EXIT'],
          context
        });
        ids.push(id);
      }

      // 2. Update outcomes
      for (let i = 0; i < ids.length; i++) {
        const success = i < 7 ? 0.85 : 0.2; // 70% success rate
        reasoningBank.updateOutcome(ids[i], { success, confidence: 0.75 });
      }

      // 3. Judge verdicts
      for (const id of ids) {
        reasoningBank.judgeVerdict(id);
      }

      // 4. Distill memory
      const patterns = reasoningBank.distillMemory(ids);

      // 5. Adapt thresholds
      const newThresholds = reasoningBank.adaptThresholds({
        successRate: 0.7,
        avgConfidence: 0.75,
        volatility: 0.15
      });

      // Verify learning occurred
      expect(patterns.length).toBe(1);
      expect(patterns[0].successRate).toBeCloseTo(0.7, 1);
      expect(newThresholds).toBeDefined();

      // 6. Query learned pattern
      const learned = reasoningBank.query(context);
      expect(learned.total).toBe(10);
      expect(learned.successes).toBe(7);
    });

    it('should improve decision making over time', () => {
      const iterations = 5;
      const resultsOverTime = [];

      for (let iter = 0; iter < iterations; iter++) {
        const ids = [];
        const context = { iteration: iter };

        // Simulate improving performance
        const baseSuccessRate = 0.5 + (iter * 0.1);

        for (let i = 0; i < 20; i++) {
          const id = reasoningBank.recordExperience({
            trajectory: ['ACTION'],
            context
          });

          const success = Math.random() < baseSuccessRate ? 0.9 : 0.1;
          reasoningBank.updateOutcome(id, { success, confidence: 0.7 });
          reasoningBank.judgeVerdict(id);
          ids.push(id);
        }

        const patterns = reasoningBank.distillMemory(ids);
        resultsOverTime.push(patterns[0].successRate);

        reasoningBank.adaptThresholds({
          successRate: patterns[0].successRate,
          avgConfidence: 0.7,
          volatility: 0.2
        });
      }

      // Success rate should improve over iterations
      expect(resultsOverTime[4]).toBeGreaterThan(resultsOverTime[0]);
    });
  });

  describe('Performance', () => {
    it('should handle 1000 experiences efficiently', () => {
      const start = performance.now();

      for (let i = 0; i < 1000; i++) {
        const id = reasoningBank.recordExperience({
          trajectory: ['ACTION_' + i],
          context: { batch: Math.floor(i / 100) }
        });

        reasoningBank.updateOutcome(id, { success: Math.random() });
        reasoningBank.judgeVerdict(id);
      }

      const duration = performance.now() - start;

      expect(duration).toBeLessThan(100);
      expect(reasoningBank.experiences.length).toBe(1000);
    });

    it('should distill large memory sets quickly', () => {
      // Create 500 experiences across 10 contexts
      const ids = [];
      for (let i = 0; i < 500; i++) {
        const id = reasoningBank.recordExperience({
          trajectory: ['TRADE'],
          context: { type: i % 10 }
        });
        reasoningBank.updateOutcome(id, { success: Math.random() });
        reasoningBank.judgeVerdict(id);
        ids.push(id);
      }

      const start = performance.now();
      const patterns = reasoningBank.distillMemory(ids);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(50);
      expect(patterns.length).toBe(10);
    });
  });
});
