/**
 * Tests for prompt builder
 */

import { PromptBuilder } from '../src/prompt-builder';

describe('PromptBuilder', () => {
  let builder: PromptBuilder;

  beforeEach(() => {
    builder = new PromptBuilder();
  });

  describe('basic message building', () => {
    it('should build messages with system, user, and assistant', () => {
      const messages = builder
        .system('You are a helpful assistant')
        .user('Hello')
        .assistant('Hi there!')
        .build();

      expect(messages).toEqual([
        { role: 'system', content: 'You are a helpful assistant' },
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there!' },
      ]);
    });

    it('should clear messages', () => {
      builder.user('Hello').clear();
      expect(() => builder.build()).toThrow('Cannot build empty prompt');
    });
  });

  describe('template usage', () => {
    it('should use predefined template', () => {
      const messages = builder
        .useTemplate('trading-analysis', {
          strategy_type: 'momentum',
          market_data: 'SPY',
          timeframe: '1h',
        })
        .build();

      expect(messages).toHaveLength(2);
      expect(messages[0].role).toBe('system');
      expect(messages[0].content).toContain('momentum');
      expect(messages[1].role).toBe('user');
      expect(messages[1].content).toContain('SPY');
    });

    it('should throw error for unknown template', () => {
      expect(() => builder.useTemplate('unknown-template')).toThrow(
        'Template "unknown-template" not found'
      );
    });
  });

  describe('advanced features', () => {
    it('should add few-shot examples', () => {
      const messages = builder
        .system('You are a helpful assistant')
        .addFewShotExamples([
          { user: 'What is 2+2?', assistant: '4' },
          { user: 'What is 3+3?', assistant: '6' },
        ])
        .user('What is 4+4?')
        .build();

      expect(messages).toHaveLength(6); // system + 2 examples + user
    });

    it('should enable chain of thought', () => {
      const messages = builder
        .user('Solve this problem')
        .enableChainOfThought()
        .build();

      expect(messages[0].content).toContain("Let's think step by step");
    });

    it('should request structured output', () => {
      const messages = builder
        .user('Generate data')
        .requestStructuredOutput('json')
        .build();

      expect(messages[0].content).toContain('JSON');
    });
  });

  describe('token estimation', () => {
    it('should estimate tokens', () => {
      builder.user('Hello world');
      const tokens = builder.estimateTokens();
      expect(tokens).toBeGreaterThan(0);
    });
  });

  describe('import/export', () => {
    it('should export messages', () => {
      builder.user('Hello');
      const exported = builder.export();
      expect(exported).toEqual([{ role: 'user', content: 'Hello' }]);
    });

    it('should import messages', () => {
      const messages = [
        { role: 'system' as const, content: 'System' },
        { role: 'user' as const, content: 'User' },
      ];

      builder.import(messages);
      expect(builder.build()).toEqual(messages);
    });
  });
});
