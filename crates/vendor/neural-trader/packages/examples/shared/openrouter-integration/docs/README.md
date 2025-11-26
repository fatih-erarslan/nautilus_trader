# OpenRouter Integration

Unified interface for multiple LLM providers with smart model routing, cost optimization, and advanced prompt engineering.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Meta, Mistral, Google models
- **Smart Model Selection**: Automatic routing based on task complexity
- **Cost Optimization**: Budget-aware model selection and cost estimation
- **Rate Limiting**: Built-in throttling and retry logic
- **Prompt Engineering**: Template system with advanced features

## Installation

```bash
npm install @neural-trader/openrouter-integration
```

## Quick Start

```typescript
import { createOpenRouterClient } from '@neural-trader/openrouter-integration';

// Create client
const { client, selector, promptBuilder } = createOpenRouterClient('your-api-key');

// Build prompt
const messages = promptBuilder
  .system('You are a trading analyst')
  .user('Analyze current market trends')
  .build();

// Select optimal model
const model = selector.selectModel({
  complexity: 'complex',
  requiresCodeGeneration: false,
  maxCost: 0.01, // USD per request
});

// Execute request
const response = await client.complete({
  model: model.id,
  messages,
  temperature: 0.7,
  maxTokens: 1000,
});

console.log(response.choices[0].message.content);
console.log('Cost:', response.cost);
```

## Model Selection

### Automatic Selection

```typescript
import { ModelSelector } from '@neural-trader/openrouter-integration';

const selector = new ModelSelector();

// Select by task requirements
const model = selector.selectModel({
  complexity: 'moderate',
  maxCost: 0.001,
  maxLatency: 2000,
  requiresReasoning: true,
  requiresCodeGeneration: true,
  contextLength: 8000,
});

console.log(`Selected: ${model.name}`);
console.log(`Cost: $${model.cost.prompt}/M prompt tokens`);
```

### Model Comparison

```typescript
// Compare models for a task
const comparison = selector.compareModels({
  complexity: 'complex',
  requiresCodeGeneration: true,
});

comparison.forEach(({ model, score, costEfficiency, reasoning }) => {
  console.log(`${model.name}: Score ${score.toFixed(2)}`);
  console.log(`  ${reasoning}`);
});
```

## Prompt Engineering

### Using Templates

```typescript
import { PromptBuilder } from '@neural-trader/openrouter-integration';

const builder = new PromptBuilder();

// Use predefined template
const messages = builder.useTemplate('trading-analysis', {
  strategy_type: 'momentum',
  market_data: 'SPY, QQQ',
  timeframe: '1-day',
}).build();
```

### Custom Templates

```typescript
builder.registerTemplate('my-template', {
  system: 'You are a {role} specialist',
  user: 'Analyze {data} and provide {output_format}',
  variables: {
    role: 'data science',
    data: 'market data',
    output_format: 'insights',
  },
});

const messages = builder.useTemplate('my-template', {
  role: 'trading',
  data: 'price movements',
  output_format: 'trading signals',
}).build();
```

### Advanced Features

```typescript
// Chain of thought reasoning
builder
  .system('You are an expert analyst')
  .user('Solve this complex problem')
  .enableChainOfThought()
  .build();

// Structured output
builder
  .user('Generate trading signals')
  .requestStructuredOutput('json')
  .build();

// Few-shot learning
builder
  .system('You are a pattern recognizer')
  .addFewShotExamples([
    { user: 'Pattern: ABC', assistant: 'Type: Triangle' },
    { user: 'Pattern: XYZ', assistant: 'Type: Head and Shoulders' },
  ])
  .user('Pattern: DEF')
  .build();
```

## Cost Management

```typescript
// Estimate cost before execution
const cost = await client.estimateCost(
  'openai/gpt-4-turbo-preview',
  1000, // prompt tokens
  500   // completion tokens
);

console.log(`Estimated cost: $${cost.totalCost}`);

// Select cost-effective model
const cheapModel = selector.selectModel({
  complexity: 'simple',
  maxCost: 0.0001, // Very low cost per token
});
```

## Rate Limiting

```typescript
// Configure rate limits
const client = new OpenRouterClient({
  apiKey: 'your-key',
  rateLimit: {
    interval: 1000, // ms
    limit: 10,      // requests per interval
  },
  maxRetries: 3,
  timeout: 30000,
});
```

## Available Models

### OpenAI
- GPT-4 Turbo: High performance, reasoning
- GPT-4: Maximum accuracy
- GPT-3.5 Turbo: Cost-effective, fast

### Anthropic
- Claude 3 Opus: Best reasoning, long context
- Claude 3 Sonnet: Balanced performance
- Claude 3 Haiku: Fast, efficient

### Meta
- Llama 3 70B: Strong open-source option
- Llama 3 8B: Very cost-effective

### Mistral
- Mixtral 8x7B: Multilingual, balanced
- Mistral 7B: Fast and affordable

### Google
- Gemini Pro: Multimodal capabilities

## API Reference

### OpenRouterClient

```typescript
class OpenRouterClient {
  constructor(config: OpenRouterConfig);
  complete(request: ModelRequest): Promise<ModelResponse>;
  getModels(): Promise<ModelInfo[]>;
  estimateCost(model: string, promptTokens: number, completionTokens: number): Promise<CostEstimate>;
}
```

### ModelSelector

```typescript
class ModelSelector {
  selectModel(requirements: TaskRequirements): ModelCapabilities;
  compareModels(requirements: TaskRequirements, modelIds?: string[]): ComparisonResult[];
  estimateModelCost(modelId: string, promptTokens: number, completionTokens: number): CostEstimate;
  getAvailableModels(): ModelCapabilities[];
}
```

### PromptBuilder

```typescript
class PromptBuilder {
  system(content: string): this;
  user(content: string): this;
  assistant(content: string): this;
  useTemplate(name: string, variables?: Record<string, string>): this;
  registerTemplate(name: string, template: PromptTemplate): void;
  enableChainOfThought(): this;
  requestStructuredOutput(format: 'json' | 'markdown' | 'code'): this;
  addFewShotExamples(examples: Array<{user: string, assistant: string}>): this;
  build(): Message[];
}
```

## Best Practices

1. **Start with cost-effective models** for simple tasks
2. **Use templates** for consistent prompt engineering
3. **Enable rate limiting** to avoid API throttling
4. **Estimate costs** before expensive operations
5. **Cache results** when appropriate
6. **Use chain-of-thought** for complex reasoning tasks
7. **Implement retry logic** for production systems

## License

MIT
