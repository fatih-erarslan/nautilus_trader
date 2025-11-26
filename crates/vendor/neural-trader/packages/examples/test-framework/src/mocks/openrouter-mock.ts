/**
 * Mock OpenRouter for testing
 */

import { MockOptions } from '../types';

export interface OpenRouterRequest {
  model: string;
  messages: Array<{ role: string; content: string }>;
  temperature?: number;
  max_tokens?: number;
}

export interface OpenRouterResponse {
  id: string;
  choices: Array<{
    message: { role: string; content: string };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

/**
 * Mock OpenRouter client
 */
export class MockOpenRouter {
  private responses: string[];
  private options: MockOptions;
  private callCount: number;
  private requests: OpenRouterRequest[];

  constructor(options: MockOptions = {}) {
    this.responses = options.responses || [
      'This is a mock response',
      'Analysis complete',
      'Prediction generated'
    ];
    this.options = options;
    this.callCount = 0;
    this.requests = [];
  }

  /**
   * Mock chat completion
   */
  async chat(request: OpenRouterRequest): Promise<OpenRouterResponse> {
    this.callCount++;
    this.requests.push(request);

    await this.simulateDelay();
    this.maybeThrowError();

    const responseIndex = this.callCount % this.responses.length;
    const content = this.responses[responseIndex];

    return {
      id: `mock-${this.callCount}`,
      choices: [
        {
          message: {
            role: 'assistant',
            content
          },
          finish_reason: 'stop'
        }
      ],
      usage: {
        prompt_tokens: this.estimateTokens(request.messages),
        completion_tokens: this.estimateTokens([{ role: 'assistant', content }]),
        total_tokens: 0
      }
    };
  }

  /**
   * Set mock responses
   */
  setResponses(responses: string[]): void {
    this.responses = responses;
  }

  /**
   * Get all requests made
   */
  getRequests(): OpenRouterRequest[] {
    return [...this.requests];
  }

  /**
   * Get call count
   */
  getCallCount(): number {
    return this.callCount;
  }

  /**
   * Reset mock state
   */
  reset(): void {
    this.callCount = 0;
    this.requests = [];
  }

  private estimateTokens(messages: Array<{ role: string; content: string }>): number {
    return messages.reduce((sum, msg) => sum + Math.ceil(msg.content.length / 4), 0);
  }

  private async simulateDelay(): Promise<void> {
    const delay = this.options.delay || 50;
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  private maybeThrowError(): void {
    if (this.options.errorRate && Math.random() < this.options.errorRate) {
      throw new Error('Mock OpenRouter error');
    }
  }
}

/**
 * Create mock OpenRouter client
 */
export function createMockOpenRouter(options: MockOptions = {}): MockOpenRouter {
  return new MockOpenRouter(options);
}
