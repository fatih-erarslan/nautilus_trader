/**
 * Smart model selector based on task complexity and cost optimization
 * Routes requests to optimal models based on requirements
 */

export type TaskComplexity = 'simple' | 'moderate' | 'complex' | 'expert';

export interface TaskRequirements {
  complexity: TaskComplexity;
  maxCost?: number; // USD
  maxLatency?: number; // ms
  requiresReasoning?: boolean;
  requiresCodeGeneration?: boolean;
  requiresMultimodal?: boolean;
  contextLength?: number; // tokens
}

export interface ModelCapabilities {
  id: string;
  name: string;
  provider: string;
  cost: {
    prompt: number; // per 1M tokens
    completion: number; // per 1M tokens
  };
  avgLatency: number; // ms
  contextLength: number;
  strengths: string[];
  bestFor: TaskComplexity[];
}

export class ModelSelector {
  private models: ModelCapabilities[] = [
    // OpenAI Models
    {
      id: 'openai/gpt-4-turbo-preview',
      name: 'GPT-4 Turbo',
      provider: 'openai',
      cost: { prompt: 10.0, completion: 30.0 },
      avgLatency: 3000,
      contextLength: 128000,
      strengths: ['reasoning', 'code', 'analysis'],
      bestFor: ['complex', 'expert'],
    },
    {
      id: 'openai/gpt-4',
      name: 'GPT-4',
      provider: 'openai',
      cost: { prompt: 30.0, completion: 60.0 },
      avgLatency: 4000,
      contextLength: 8192,
      strengths: ['reasoning', 'creativity', 'accuracy'],
      bestFor: ['expert'],
    },
    {
      id: 'openai/gpt-3.5-turbo',
      name: 'GPT-3.5 Turbo',
      provider: 'openai',
      cost: { prompt: 0.5, completion: 1.5 },
      avgLatency: 1000,
      contextLength: 16385,
      strengths: ['speed', 'cost-effective'],
      bestFor: ['simple', 'moderate'],
    },

    // Anthropic Models
    {
      id: 'anthropic/claude-3-opus',
      name: 'Claude 3 Opus',
      provider: 'anthropic',
      cost: { prompt: 15.0, completion: 75.0 },
      avgLatency: 3500,
      contextLength: 200000,
      strengths: ['reasoning', 'analysis', 'code', 'long-context'],
      bestFor: ['complex', 'expert'],
    },
    {
      id: 'anthropic/claude-3-sonnet',
      name: 'Claude 3 Sonnet',
      provider: 'anthropic',
      cost: { prompt: 3.0, completion: 15.0 },
      avgLatency: 2000,
      contextLength: 200000,
      strengths: ['balanced', 'code', 'reasoning'],
      bestFor: ['moderate', 'complex'],
    },
    {
      id: 'anthropic/claude-3-haiku',
      name: 'Claude 3 Haiku',
      provider: 'anthropic',
      cost: { prompt: 0.25, completion: 1.25 },
      avgLatency: 800,
      contextLength: 200000,
      strengths: ['speed', 'cost-effective', 'long-context'],
      bestFor: ['simple', 'moderate'],
    },

    // Meta Models
    {
      id: 'meta-llama/llama-3-70b-instruct',
      name: 'Llama 3 70B',
      provider: 'meta',
      cost: { prompt: 0.59, completion: 0.79 },
      avgLatency: 1500,
      contextLength: 8192,
      strengths: ['open-source', 'cost-effective', 'code'],
      bestFor: ['moderate', 'complex'],
    },
    {
      id: 'meta-llama/llama-3-8b-instruct',
      name: 'Llama 3 8B',
      provider: 'meta',
      cost: { prompt: 0.06, completion: 0.06 },
      avgLatency: 500,
      contextLength: 8192,
      strengths: ['speed', 'very-cost-effective'],
      bestFor: ['simple'],
    },

    // Mistral Models
    {
      id: 'mistralai/mixtral-8x7b-instruct',
      name: 'Mixtral 8x7B',
      provider: 'mistral',
      cost: { prompt: 0.24, completion: 0.24 },
      avgLatency: 1200,
      contextLength: 32768,
      strengths: ['balanced', 'multilingual', 'cost-effective'],
      bestFor: ['moderate', 'complex'],
    },
    {
      id: 'mistralai/mistral-7b-instruct',
      name: 'Mistral 7B',
      provider: 'mistral',
      cost: { prompt: 0.06, completion: 0.06 },
      avgLatency: 600,
      contextLength: 32768,
      strengths: ['speed', 'cost-effective'],
      bestFor: ['simple', 'moderate'],
    },

    // Google Models
    {
      id: 'google/gemini-pro',
      name: 'Gemini Pro',
      provider: 'google',
      cost: { prompt: 0.125, completion: 0.375 },
      avgLatency: 1800,
      contextLength: 32768,
      strengths: ['multimodal', 'reasoning', 'code'],
      bestFor: ['moderate', 'complex'],
    },
  ];

  /**
   * Select the best model based on task requirements
   */
  selectModel(requirements: TaskRequirements): ModelCapabilities {
    let candidates = this.models.filter((model) =>
      model.bestFor.includes(requirements.complexity)
    );

    // Filter by cost constraint
    if (requirements.maxCost !== undefined) {
      candidates = candidates.filter((model) => {
        const avgCost = (model.cost.prompt + model.cost.completion) / 2;
        return avgCost <= requirements.maxCost * 1_000_000; // Convert to per token
      });
    }

    // Filter by latency constraint
    if (requirements.maxLatency !== undefined) {
      candidates = candidates.filter(
        (model) => model.avgLatency <= requirements.maxLatency
      );
    }

    // Filter by context length
    if (requirements.contextLength !== undefined) {
      candidates = candidates.filter(
        (model) => model.contextLength >= requirements.contextLength
      );
    }

    // Filter by capabilities
    if (requirements.requiresReasoning) {
      candidates = candidates.filter((model) =>
        model.strengths.includes('reasoning')
      );
    }

    if (requirements.requiresCodeGeneration) {
      candidates = candidates.filter((model) =>
        model.strengths.includes('code')
      );
    }

    if (requirements.requiresMultimodal) {
      candidates = candidates.filter((model) =>
        model.strengths.includes('multimodal')
      );
    }

    if (candidates.length === 0) {
      throw new Error(
        'No models match the specified requirements. Consider relaxing constraints.'
      );
    }

    // Select best model: optimize for cost-effectiveness at same complexity level
    return this.optimizeSelection(candidates, requirements);
  }

  /**
   * Get all available models
   */
  getAvailableModels(): ModelCapabilities[] {
    return [...this.models];
  }

  /**
   * Get models by provider
   */
  getModelsByProvider(provider: string): ModelCapabilities[] {
    return this.models.filter((model) => model.provider === provider);
  }

  /**
   * Get cost estimate for a specific model
   */
  estimateModelCost(
    modelId: string,
    promptTokens: number,
    completionTokens: number
  ): { promptCost: number; completionCost: number; totalCost: number } {
    const model = this.models.find((m) => m.id === modelId);
    if (!model) {
      throw new Error(`Model ${modelId} not found`);
    }

    const promptCost = (promptTokens / 1_000_000) * model.cost.prompt;
    const completionCost = (completionTokens / 1_000_000) * model.cost.completion;

    return {
      promptCost,
      completionCost,
      totalCost: promptCost + completionCost,
    };
  }

  /**
   * Compare multiple models for a given task
   */
  compareModels(
    requirements: TaskRequirements,
    modelIds?: string[]
  ): Array<{
    model: ModelCapabilities;
    score: number;
    costEfficiency: number;
    reasoning: string;
  }> {
    const candidates = modelIds
      ? this.models.filter((m) => modelIds.includes(m.id))
      : this.models.filter((m) => m.bestFor.includes(requirements.complexity));

    return candidates.map((model) => {
      const score = this.calculateScore(model, requirements);
      const costEfficiency = this.calculateCostEfficiency(model);
      const reasoning = this.generateReasoning(model, requirements);

      return { model, score, costEfficiency, reasoning };
    }).sort((a, b) => b.score - a.score);
  }

  /**
   * Optimize model selection based on requirements
   */
  private optimizeSelection(
    candidates: ModelCapabilities[],
    requirements: TaskRequirements
  ): ModelCapabilities {
    // Calculate score for each candidate
    const scored = candidates.map((model) => ({
      model,
      score: this.calculateScore(model, requirements),
    }));

    // Sort by score (higher is better)
    scored.sort((a, b) => b.score - a.score);

    return scored[0].model;
  }

  /**
   * Calculate model score based on requirements
   */
  private calculateScore(
    model: ModelCapabilities,
    requirements: TaskRequirements
  ): number {
    let score = 0;

    // Base score for complexity match
    if (model.bestFor.includes(requirements.complexity)) {
      score += 100;
    }

    // Cost efficiency (inverse of average cost per token)
    const avgCost = (model.cost.prompt + model.cost.completion) / 2;
    score += (1 / avgCost) * 10;

    // Latency bonus (inverse of latency)
    score += (1 / model.avgLatency) * 1000;

    // Context length bonus
    score += Math.log(model.contextLength) * 5;

    // Strength matching
    if (requirements.requiresReasoning && model.strengths.includes('reasoning')) {
      score += 20;
    }
    if (requirements.requiresCodeGeneration && model.strengths.includes('code')) {
      score += 20;
    }
    if (requirements.requiresMultimodal && model.strengths.includes('multimodal')) {
      score += 20;
    }

    return score;
  }

  /**
   * Calculate cost efficiency metric
   */
  private calculateCostEfficiency(model: ModelCapabilities): number {
    const avgCost = (model.cost.prompt + model.cost.completion) / 2;
    return 1 / avgCost; // Higher is more efficient
  }

  /**
   * Generate reasoning for model selection
   */
  private generateReasoning(
    model: ModelCapabilities,
    requirements: TaskRequirements
  ): string {
    const reasons: string[] = [];

    if (model.bestFor.includes(requirements.complexity)) {
      reasons.push(`Optimized for ${requirements.complexity} tasks`);
    }

    const avgCost = (model.cost.prompt + model.cost.completion) / 2;
    if (avgCost < 1.0) {
      reasons.push('Cost-effective');
    }

    if (model.avgLatency < 1000) {
      reasons.push('Low latency');
    }

    if (model.contextLength > 30000) {
      reasons.push('Large context window');
    }

    model.strengths.forEach((strength) => {
      if (
        (requirements.requiresReasoning && strength === 'reasoning') ||
        (requirements.requiresCodeGeneration && strength === 'code') ||
        (requirements.requiresMultimodal && strength === 'multimodal')
      ) {
        reasons.push(`Strong ${strength} capabilities`);
      }
    });

    return reasons.join(', ');
  }
}
