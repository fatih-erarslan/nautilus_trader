/**
 * OpenRouter API client with rate limiting and retry logic
 * Provides unified interface for multiple LLM providers
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import pRetry from 'p-retry';
import { throttle } from 'p-throttle';

export interface OpenRouterConfig {
  apiKey: string;
  baseURL?: string;
  timeout?: number;
  maxRetries?: number;
  rateLimit?: {
    interval: number; // ms
    limit: number; // requests per interval
  };
}

export interface ModelRequest {
  model: string;
  messages: Array<{
    role: 'system' | 'user' | 'assistant';
    content: string;
  }>;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  stream?: boolean;
}

export interface ModelResponse {
  id: string;
  model: string;
  choices: Array<{
    message: {
      role: string;
      content: string;
    };
    finishReason: string;
  }>;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  cost?: {
    promptCost: number;
    completionCost: number;
    totalCost: number;
  };
}

export class OpenRouterClient {
  private client: AxiosInstance;
  private config: Required<OpenRouterConfig>;
  private throttledRequest: ReturnType<typeof throttle>;

  constructor(config: OpenRouterConfig) {
    this.config = {
      apiKey: config.apiKey,
      baseURL: config.baseURL || 'https://openrouter.ai/api/v1',
      timeout: config.timeout || 60000,
      maxRetries: config.maxRetries || 3,
      rateLimit: config.rateLimit || { interval: 1000, limit: 10 },
    };

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        Authorization: `Bearer ${this.config.apiKey}`,
        'HTTP-Referer': 'https://neural-trader.ai',
        'X-Title': 'Neural Trader',
      },
    });

    // Create throttled request function
    this.throttledRequest = throttle({
      interval: this.config.rateLimit.interval,
      limit: this.config.rateLimit.limit,
    })<typeof this.makeRequest.bind(this)>(this.makeRequest.bind(this));
  }

  /**
   * Complete a chat request with automatic retry and rate limiting
   */
  async complete(request: ModelRequest): Promise<ModelResponse> {
    return pRetry(
      async () => {
        return this.throttledRequest(request);
      },
      {
        retries: this.config.maxRetries,
        onFailedAttempt: (error) => {
          console.warn(
            `Request failed (attempt ${error.attemptNumber}/${this.config.maxRetries}):`,
            error.message
          );
        },
      }
    );
  }

  /**
   * Make the actual API request
   */
  private async makeRequest(request: ModelRequest): Promise<ModelResponse> {
    try {
      const response = await this.client.post('/chat/completions', {
        model: request.model,
        messages: request.messages,
        temperature: request.temperature ?? 0.7,
        max_tokens: request.maxTokens ?? 1000,
        top_p: request.topP,
        frequency_penalty: request.frequencyPenalty,
        presence_penalty: request.presencePenalty,
        stream: request.stream ?? false,
      });

      return this.transformResponse(response.data);
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw this.handleError(error);
      }
      throw error;
    }
  }

  /**
   * Get available models with pricing information
   */
  async getModels(): Promise<Array<{
    id: string;
    name: string;
    pricing: {
      prompt: number;
      completion: number;
    };
    contextLength: number;
  }>> {
    try {
      const response = await this.client.get('/models');
      return response.data.data.map((model: any) => ({
        id: model.id,
        name: model.name,
        pricing: {
          prompt: parseFloat(model.pricing.prompt),
          completion: parseFloat(model.pricing.completion),
        },
        contextLength: model.context_length,
      }));
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw this.handleError(error);
      }
      throw error;
    }
  }

  /**
   * Transform OpenRouter response to standard format
   */
  private transformResponse(data: any): ModelResponse {
    return {
      id: data.id,
      model: data.model,
      choices: data.choices.map((choice: any) => ({
        message: {
          role: choice.message.role,
          content: choice.message.content,
        },
        finishReason: choice.finish_reason,
      })),
      usage: {
        promptTokens: data.usage.prompt_tokens,
        completionTokens: data.usage.completion_tokens,
        totalTokens: data.usage.total_tokens,
      },
      cost: data.usage.prompt_cost
        ? {
            promptCost: data.usage.prompt_cost,
            completionCost: data.usage.completion_cost,
            totalCost: data.usage.total_cost,
          }
        : undefined,
    };
  }

  /**
   * Handle API errors with detailed messages
   */
  private handleError(error: AxiosError): Error {
    const status = error.response?.status;
    const message = (error.response?.data as any)?.error?.message || error.message;

    switch (status) {
      case 400:
        return new Error(`Bad Request: ${message}`);
      case 401:
        return new Error(`Authentication failed: ${message}`);
      case 402:
        return new Error(`Insufficient credits: ${message}`);
      case 429:
        return new Error(`Rate limit exceeded: ${message}`);
      case 500:
      case 502:
      case 503:
        return new Error(`OpenRouter service error: ${message}`);
      default:
        return new Error(`OpenRouter request failed: ${message}`);
    }
  }

  /**
   * Estimate cost for a request before executing
   */
  async estimateCost(
    model: string,
    promptTokens: number,
    completionTokens: number
  ): Promise<{
    promptCost: number;
    completionCost: number;
    totalCost: number;
  }> {
    const models = await this.getModels();
    const modelInfo = models.find((m) => m.id === model);

    if (!modelInfo) {
      throw new Error(`Model ${model} not found`);
    }

    const promptCost = (promptTokens / 1_000_000) * modelInfo.pricing.prompt;
    const completionCost = (completionTokens / 1_000_000) * modelInfo.pricing.completion;

    return {
      promptCost,
      completionCost,
      totalCost: promptCost + completionCost,
    };
  }
}
