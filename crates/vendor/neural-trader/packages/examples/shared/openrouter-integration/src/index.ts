/**
 * OpenRouter Integration Package
 *
 * Provides unified interface for multiple LLM providers with:
 * - Smart model routing based on task complexity
 * - Cost optimization and budget constraints
 * - Rate limiting and retry logic
 * - Prompt engineering utilities
 *
 * @packageDocumentation
 */

export { OpenRouterClient } from './client';
export type { OpenRouterConfig, ModelRequest, ModelResponse } from './client';

export { ModelSelector } from './model-selector';
export type { TaskComplexity, TaskRequirements, ModelCapabilities } from './model-selector';

export { PromptBuilder } from './prompt-builder';
export type { PromptTemplate, Message } from './prompt-builder';

/**
 * Factory function to create a complete OpenRouter setup
 */
export function createOpenRouterClient(apiKey: string) {
  const client = new OpenRouterClient({ apiKey });
  const selector = new ModelSelector();
  const promptBuilder = new PromptBuilder();

  return {
    client,
    selector,
    promptBuilder,

    /**
     * Helper: Complete a request with automatic model selection
     */
    async completeWithAutoSelection(
      requirements: import('./model-selector').TaskRequirements,
      messages: import('./prompt-builder').Message[]
    ) {
      const model = selector.selectModel(requirements);
      return client.complete({
        model: model.id,
        messages,
      });
    },

    /**
     * Helper: Get cost estimate for a prompt
     */
    async estimateCost(
      requirements: import('./model-selector').TaskRequirements,
      promptTokens: number,
      completionTokens: number
    ) {
      const model = selector.selectModel(requirements);
      return selector.estimateModelCost(model.id, promptTokens, completionTokens);
    },
  };
}
