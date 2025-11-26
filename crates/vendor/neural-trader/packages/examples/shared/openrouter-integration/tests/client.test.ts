/**
 * Tests for OpenRouter client
 */

import { OpenRouterClient } from '../src/client';

describe('OpenRouterClient', () => {
  let client: OpenRouterClient;

  beforeEach(() => {
    client = new OpenRouterClient({
      apiKey: 'test-key',
      rateLimit: { interval: 100, limit: 5 },
    });
  });

  describe('constructor', () => {
    it('should create client with default config', () => {
      expect(client).toBeInstanceOf(OpenRouterClient);
    });

    it('should create client with custom config', () => {
      const customClient = new OpenRouterClient({
        apiKey: 'test-key',
        baseURL: 'https://custom.api.com',
        timeout: 5000,
        maxRetries: 5,
      });

      expect(customClient).toBeInstanceOf(OpenRouterClient);
    });
  });

  describe('estimateCost', () => {
    it('should calculate cost correctly', async () => {
      // Mock getModels response
      jest.spyOn(client as any, 'getModels').mockResolvedValue([
        {
          id: 'test-model',
          name: 'Test Model',
          pricing: { prompt: 1.0, completion: 2.0 },
          contextLength: 4096,
        },
      ]);

      const cost = await client.estimateCost('test-model', 1000, 500);

      expect(cost).toEqual({
        promptCost: 0.001, // 1000 / 1M * 1.0
        completionCost: 0.001, // 500 / 1M * 2.0
        totalCost: 0.002,
      });
    });

    it('should throw error for unknown model', async () => {
      jest.spyOn(client as any, 'getModels').mockResolvedValue([]);

      await expect(client.estimateCost('unknown', 1000, 500)).rejects.toThrow(
        'Model unknown not found'
      );
    });
  });
});
