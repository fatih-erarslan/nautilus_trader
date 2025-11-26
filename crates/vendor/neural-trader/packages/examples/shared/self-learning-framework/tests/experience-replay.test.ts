/**
 * Tests for experience replay
 */

import { ExperienceReplay } from '../src/experience-replay';

describe('ExperienceReplay', () => {
  let replay: ExperienceReplay;

  beforeEach(() => {
    replay = new ExperienceReplay({
      maxSize: 10,
      prioritization: 'uniform',
      dbPath: ':memory:',
    });
  });

  afterEach(async () => {
    await replay.clear();
  });

  describe('store', () => {
    it('should store single experience', async () => {
      const experience = {
        id: 'exp-1',
        timestamp: new Date(),
        state: { value: 1 },
        action: { type: 'buy' },
        result: { executed: true },
        reward: 10,
      };

      await replay.store(experience);

      const stats = replay.getStats();
      expect(stats.size).toBe(1);
    });

    it('should evict oldest when buffer is full', async () => {
      // Fill buffer
      for (let i = 0; i < 12; i++) {
        await replay.store({
          id: `exp-${i}`,
          timestamp: new Date(Date.now() + i * 1000),
          state: { value: i },
          action: { type: 'action' },
          result: { value: i },
          reward: i,
        });
      }

      const stats = replay.getStats();
      expect(stats.size).toBe(10); // maxSize
    });
  });

  describe('sample', () => {
    it('should sample random experiences', async () => {
      // Add experiences
      for (let i = 0; i < 5; i++) {
        await replay.store({
          id: `exp-${i}`,
          timestamp: new Date(),
          state: { value: i },
          action: { type: 'action' },
          result: { value: i },
          reward: i,
        });
      }

      const batch = await replay.sample(3);

      expect(batch.experiences.length).toBeLessThanOrEqual(3);
      expect(batch.indices.length).toBe(batch.experiences.length);
    });
  });

  describe('querySimilar', () => {
    it('should find similar experiences', async () => {
      const baseExperience = {
        id: 'base',
        timestamp: new Date(),
        state: { value: 50 },
        action: { type: 'buy' },
        result: { executed: true },
        reward: 10,
      };

      await replay.store(baseExperience);

      // Add similar experiences
      for (let i = 45; i <= 55; i++) {
        await replay.store({
          id: `exp-${i}`,
          timestamp: new Date(),
          state: { value: i },
          action: { type: 'buy' },
          result: { executed: true },
          reward: i,
        });
      }

      const similar = await replay.querySimilar(baseExperience, 5);

      expect(similar.length).toBeGreaterThan(0);
      expect(similar.length).toBeLessThanOrEqual(5);
    });
  });

  describe('getHighReward', () => {
    it('should filter by reward threshold', async () => {
      for (let i = 0; i < 10; i++) {
        await replay.store({
          id: `exp-${i}`,
          timestamp: new Date(),
          state: { value: i },
          action: { type: 'action' },
          result: { value: i },
          reward: i * 10,
        });
      }

      const highReward = await replay.getHighReward(50);

      expect(highReward.every(exp => exp.reward >= 50)).toBe(true);
    });
  });
});
