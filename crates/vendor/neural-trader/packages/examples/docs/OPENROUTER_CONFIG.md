# OpenRouter Configuration Guide

Complete guide to integrating OpenRouter for AI-powered insights.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Rate Limiting](#rate-limiting)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)
- [Cost Optimization](#cost-optimization)

---

## Introduction

OpenRouter provides access to multiple LLM providers through a unified API. Neural Trader examples use OpenRouter for:

- Strategy recommendations
- Anomaly explanations
- Feature importance analysis
- Risk assessment narratives
- Parameter optimization suggestions

**Benefits**:
- Access to Claude, GPT-4, Llama, and more
- Automatic fallback between providers
- Usage tracking and cost control
- No separate API keys needed

---

## Setup

### 1. Get API Key

```bash
# Sign up at https://openrouter.ai
# Get your API key from dashboard
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

### 2. Install Dependencies

```bash
npm install openai@^4.0.0
```

### 3. Configure in Code

```typescript
import { OpenAI } from 'openai';

const openrouter = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    'HTTP-Referer': 'https://your-app.com',
    'X-Title': 'Neural Trader'
  }
});
```

---

## Basic Usage

### Simple Query

```typescript
async function getAIInsight(prompt: string): Promise<string> {
  const response = await openrouter.chat.completions.create({
    model: 'anthropic/claude-3.5-sonnet',
    messages: [
      {
        role: 'user',
        content: prompt
      }
    ]
  });

  return response.choices[0].message.content;
}

// Usage
const insight = await getAIInsight(
  'Analyze this trading pattern: high volume, narrow spreads, increasing prices'
);
```

### Model Selection

```typescript
// Available models
const models = {
  // Claude models (best reasoning)
  claude: 'anthropic/claude-3.5-sonnet',
  claudeHaiku: 'anthropic/claude-3-haiku',  // Faster, cheaper

  // GPT models
  gpt4: 'openai/gpt-4-turbo-preview',
  gpt35: 'openai/gpt-3.5-turbo',

  // Open source
  llama: 'meta-llama/llama-3-70b-instruct',
  mixtral: 'mistralai/mixtral-8x7b-instruct',

  // Specialized
  codeLlama: 'codellama/codellama-70b-instruct'
};

// Use specific model
const response = await openrouter.chat.completions.create({
  model: models.claude,
  messages: [...]
});
```

---

## Advanced Features

### Anomaly Explanation

```typescript
export class AIAnomalyExplainer {
  constructor(private openrouter: OpenAI) {}

  async explainAnomaly(
    data: any,
    anomalyScore: number,
    context: string
  ): Promise<string> {
    const prompt = `
You are an expert financial analyst. Analyze this anomaly:

Context: ${context}
Anomaly Score: ${anomalyScore}
Data: ${JSON.stringify(data, null, 2)}

Provide a concise explanation of:
1. What makes this anomalous?
2. Possible causes
3. Risk assessment
4. Recommended actions

Format as markdown.
`;

    const response = await this.openrouter.chat.completions.create({
      model: 'anthropic/claude-3.5-sonnet',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: 500
    });

    return response.choices[0].message.content;
  }
}

// Usage
const explainer = new AIAnomalyExplainer(openrouter);

const explanation = await explainer.explainAnomaly(
  metrics,
  0.95,
  'BTC/USD market during high volatility'
);

console.log(explanation);
```

### Strategy Recommendations

```typescript
export class AIStrategyAdvisor {
  async recommendStrategy(
    marketConditions: MarketConditions,
    availableStrategies: string[],
    historicalPerformance: Record<string, PerformanceMetrics>
  ): Promise<StrategyRecommendation> {
    const prompt = `
Analyze these market conditions and recommend the best trading strategy:

Market Conditions:
- Volatility: ${marketConditions.volatility}
- Trend: ${marketConditions.trend}
- Volume: ${marketConditions.volume}
- Liquidity: ${marketConditions.liquidity}

Available Strategies: ${availableStrategies.join(', ')}

Historical Performance:
${JSON.stringify(historicalPerformance, null, 2)}

Provide:
1. Recommended strategy
2. Confidence level (0-1)
3. Key reasoning points
4. Risk warnings

Respond in JSON format:
{
  "strategy": "strategy_name",
  "confidence": 0.0-1.0,
  "reasoning": ["point1", "point2", "point3"],
  "risks": ["risk1", "risk2"]
}
`;

    const response = await this.openrouter.chat.completions.create({
      model: 'anthropic/claude-3.5-sonnet',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.3,  // Lower for more deterministic
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content);
  }
}
```

### Feature Importance Analysis

```typescript
export class AIFeatureAnalyzer {
  async analyzeFeatureImportance(
    features: Record<string, number>,
    outcome: Outcome,
    context: string
  ): Promise<FeatureAnalysis> {
    const prompt = `
Analyze which features contributed most to this outcome:

Features: ${JSON.stringify(features, null, 2)}
Outcome: ${JSON.stringify(outcome, null, 2)}
Context: ${context}

For each feature:
1. Rate importance (0-1)
2. Explain contribution
3. Suggest improvements

Return JSON:
{
  "features": [
    {
      "name": "feature_name",
      "importance": 0.0-1.0,
      "contribution": "explanation",
      "suggestion": "improvement"
    }
  ],
  "insights": ["insight1", "insight2"]
}
`;

    const response = await this.openrouter.chat.completions.create({
      model: 'anthropic/claude-3.5-sonnet',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.5,
      response_format: { type: 'json_object' }
    });

    return JSON.parse(response.choices[0].message.content);
  }
}
```

### Portfolio Narrative

```typescript
export class AIPortfolioNarrator {
  async generateReport(
    portfolio: Portfolio,
    performance: PerformanceMetrics,
    marketContext: MarketContext
  ): Promise<string> {
    const prompt = `
Generate a professional portfolio analysis report:

Portfolio:
- Weights: ${JSON.stringify(portfolio.weights)}
- Assets: ${portfolio.assets.join(', ')}

Performance:
- Return: ${(performance.return * 100).toFixed(2)}%
- Risk: ${(performance.risk * 100).toFixed(2)}%
- Sharpe Ratio: ${performance.sharpeRatio.toFixed(2)}

Market Context:
- Volatility Regime: ${marketContext.regime}
- Economic Indicators: ${JSON.stringify(marketContext.indicators)}

Provide:
1. Executive summary (2-3 sentences)
2. Risk assessment
3. Performance attribution
4. Recommendations

Format as professional markdown report.
`;

    const response = await this.openrouter.chat.completions.create({
      model: 'anthropic/claude-3.5-sonnet',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.6,
      max_tokens: 1000
    });

    return response.choices[0].message.content;
  }
}
```

---

## Rate Limiting

### Basic Rate Limiter

```typescript
export class RateLimiter {
  private requests: number[] = [];
  private limit: number;
  private window: number;

  constructor(
    limit: number = 20,      // requests
    window: number = 60000   // per minute
  ) {
    this.limit = limit;
    this.window = window;
  }

  async checkLimit(): Promise<void> {
    const now = Date.now();

    // Remove old requests
    this.requests = this.requests.filter(time => now - time < this.window);

    // Wait if limit reached
    if (this.requests.length >= this.limit) {
      const oldestRequest = Math.min(...this.requests);
      const waitTime = this.window - (now - oldestRequest) + 100;

      console.log(`Rate limit reached. Waiting ${waitTime}ms...`);
      await this.delay(waitTime);
    }

    this.requests.push(now);
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Usage
const limiter = new RateLimiter(20, 60000);

async function callOpenRouter(prompt: string): Promise<string> {
  await limiter.checkLimit();

  const response = await openrouter.chat.completions.create({
    model: 'anthropic/claude-3.5-sonnet',
    messages: [{ role: 'user', content: prompt }]
  });

  return response.choices[0].message.content;
}
```

### Token-Based Rate Limiting

```typescript
export class TokenRateLimiter {
  private tokensUsed: number = 0;
  private tokenLimit: number;
  private resetTime: number;

  constructor(
    tokenLimit: number = 100000,  // tokens per day
    resetInterval: number = 86400000  // 24 hours
  ) {
    this.tokenLimit = tokenLimit;
    this.resetTime = Date.now() + resetInterval;

    // Auto-reset
    setInterval(() => {
      this.tokensUsed = 0;
      this.resetTime = Date.now() + resetInterval;
    }, resetInterval);
  }

  async checkTokens(estimatedTokens: number): Promise<void> {
    if (this.tokensUsed + estimatedTokens > this.tokenLimit) {
      const waitTime = this.resetTime - Date.now();
      throw new Error(`Token limit reached. Resets in ${waitTime}ms`);
    }

    this.tokensUsed += estimatedTokens;
  }

  recordUsage(actualTokens: number): void {
    this.tokensUsed += actualTokens;
  }

  getRemaining(): number {
    return this.tokenLimit - this.tokensUsed;
  }
}
```

---

## Error Handling

### Retry Logic

```typescript
export class ResilientOpenRouter {
  constructor(
    private openrouter: OpenAI,
    private maxRetries: number = 3
  ) {}

  async chat(
    prompt: string,
    model: string = 'anthropic/claude-3.5-sonnet'
  ): Promise<string> {
    let lastError: Error;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const response = await this.openrouter.chat.completions.create({
          model,
          messages: [{ role: 'user', content: prompt }],
          timeout: 30000
        });

        return response.choices[0].message.content;

      } catch (error) {
        lastError = error;

        // Don't retry on certain errors
        if (error.status === 401) {
          throw new Error('Invalid API key');
        }

        if (error.status === 429) {
          // Rate limited - wait longer
          const waitTime = Math.pow(2, attempt) * 2000;
          console.warn(`Rate limited. Waiting ${waitTime}ms...`);
          await this.delay(waitTime);
          continue;
        }

        // Exponential backoff
        const waitTime = Math.pow(2, attempt) * 1000;
        console.warn(`Attempt ${attempt + 1} failed. Retrying in ${waitTime}ms...`);
        await this.delay(waitTime);
      }
    }

    throw new Error(`Failed after ${this.maxRetries} attempts: ${lastError.message}`);
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

### Fallback Models

```typescript
export class FallbackOpenRouter {
  private models = [
    'anthropic/claude-3.5-sonnet',    // Primary
    'openai/gpt-4-turbo-preview',     // Fallback 1
    'meta-llama/llama-3-70b-instruct' // Fallback 2
  ];

  async chat(prompt: string): Promise<string> {
    let lastError: Error;

    for (const model of this.models) {
      try {
        console.log(`Trying model: ${model}`);

        const response = await this.openrouter.chat.completions.create({
          model,
          messages: [{ role: 'user', content: prompt }],
          timeout: 30000
        });

        return response.choices[0].message.content;

      } catch (error) {
        lastError = error;
        console.warn(`Model ${model} failed:`, error.message);
        continue;
      }
    }

    throw new Error(`All models failed. Last error: ${lastError.message}`);
  }
}
```

---

## Best Practices

### 1. Use System Messages

```typescript
const response = await openrouter.chat.completions.create({
  model: 'anthropic/claude-3.5-sonnet',
  messages: [
    {
      role: 'system',
      content: 'You are a financial analyst specializing in quantitative trading. Provide precise, data-driven insights.'
    },
    {
      role: 'user',
      content: prompt
    }
  ]
});
```

### 2. Temperature Control

```typescript
// Deterministic (analysis, classification)
temperature: 0.0-0.3

// Balanced (recommendations)
temperature: 0.5-0.7

// Creative (brainstorming)
temperature: 0.8-1.0
```

### 3. Token Optimization

```typescript
// Use shorter models for simple tasks
const simpleQuery = await openrouter.chat.completions.create({
  model: 'anthropic/claude-3-haiku',  // Faster, cheaper
  messages: [{ role: 'user', content: 'Is this bullish or bearish?' }],
  max_tokens: 50  // Limit response length
});

// Use larger models for complex analysis
const complexAnalysis = await openrouter.chat.completions.create({
  model: 'anthropic/claude-3.5-sonnet',
  messages: [{ role: 'user', content: longPrompt }],
  max_tokens: 2000
});
```

### 4. Structured Output

```typescript
// Request JSON format
const response = await openrouter.chat.completions.create({
  model: 'anthropic/claude-3.5-sonnet',
  messages: [
    {
      role: 'user',
      content: 'Analyze this data and return JSON with keys: trend, confidence, risks'
    }
  ],
  response_format: { type: 'json_object' }
});

const parsed = JSON.parse(response.choices[0].message.content);
```

### 5. Caching

```typescript
export class CachedOpenRouter {
  private cache = new Map<string, string>();

  async chat(prompt: string): Promise<string> {
    const cacheKey = this.hashPrompt(prompt);

    if (this.cache.has(cacheKey)) {
      console.log('Cache hit');
      return this.cache.get(cacheKey)!;
    }

    const response = await this.callOpenRouter(prompt);
    this.cache.set(cacheKey, response);

    // Limit cache size
    if (this.cache.size > 100) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    return response;
  }

  private hashPrompt(prompt: string): string {
    // Simple hash for demo - use crypto.createHash for production
    return prompt.slice(0, 100);
  }
}
```

---

## Cost Optimization

### 1. Model Selection

```typescript
// Cost comparison (per 1M tokens)
const costs = {
  'anthropic/claude-3-haiku': { input: 0.25, output: 1.25 },        // Cheapest
  'openai/gpt-3.5-turbo': { input: 0.50, output: 1.50 },
  'anthropic/claude-3.5-sonnet': { input: 3.00, output: 15.00 },
  'openai/gpt-4-turbo-preview': { input: 10.00, output: 30.00 }     // Most expensive
};

// Use cheaper models when possible
function selectModel(complexity: 'simple' | 'medium' | 'complex'): string {
  switch (complexity) {
    case 'simple': return 'anthropic/claude-3-haiku';
    case 'medium': return 'openai/gpt-3.5-turbo';
    case 'complex': return 'anthropic/claude-3.5-sonnet';
  }
}
```

### 2. Batch Requests

```typescript
// ❌ BAD: Multiple individual requests
for (const item of items) {
  const analysis = await analyzeItem(item);
}

// ✅ GOOD: Single batch request
const batchPrompt = `
Analyze these items and return JSON array:
${JSON.stringify(items)}

For each item provide: trend, confidence, risk.
`;

const response = await openrouter.chat.completions.create({
  model: 'anthropic/claude-3-haiku',
  messages: [{ role: 'user', content: batchPrompt }]
});

const analyses = JSON.parse(response.choices[0].message.content);
```

### 3. Usage Tracking

```typescript
export class UsageTracker {
  private totalCost = 0;
  private requestCount = 0;

  trackUsage(model: string, inputTokens: number, outputTokens: number): void {
    const costs = this.getModelCosts(model);

    const cost = (
      (inputTokens / 1000000) * costs.input +
      (outputTokens / 1000000) * costs.output
    );

    this.totalCost += cost;
    this.requestCount++;

    console.log(`Request #${this.requestCount}: $${cost.toFixed(4)}`);
    console.log(`Total cost: $${this.totalCost.toFixed(4)}`);
  }

  getStats(): UsageStats {
    return {
      totalCost: this.totalCost,
      requestCount: this.requestCount,
      avgCost: this.totalCost / this.requestCount
    };
  }
}
```

---

## Complete Example

```typescript
import { OpenAI } from 'openai';

export class AIEnhancedSwarm {
  private openrouter: OpenAI;
  private rateLimiter: RateLimiter;
  private usageTracker: UsageTracker;

  constructor() {
    this.openrouter = new OpenAI({
      baseURL: 'https://openrouter.ai/api/v1',
      apiKey: process.env.OPENROUTER_API_KEY
    });

    this.rateLimiter = new RateLimiter(20, 60000);
    this.usageTracker = new UsageTracker();
  }

  async detectAnomalyWithExplanation(
    metrics: Metrics,
    swarmConsensus: SwarmResult
  ): Promise<AnomalyResult> {
    await this.rateLimiter.checkLimit();

    const prompt = `
Analyze this potential anomaly detected by our swarm intelligence system:

Metrics: ${JSON.stringify(metrics, null, 2)}
Swarm Consensus: ${swarmConsensus.confidence} (${swarmConsensus.votesFor}/${swarmConsensus.totalVotes} agents agree)

Provide:
1. Is this truly anomalous? (yes/no)
2. Confidence level (0-1)
3. Type of anomaly
4. Detailed explanation
5. Risk assessment
6. Recommended actions

Format as JSON.
`;

    const response = await this.openrouter.chat.completions.create({
      model: 'anthropic/claude-3.5-sonnet',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.3,
      response_format: { type: 'json_object' }
    });

    this.usageTracker.trackUsage(
      'anthropic/claude-3.5-sonnet',
      response.usage.prompt_tokens,
      response.usage.completion_tokens
    );

    return JSON.parse(response.choices[0].message.content);
  }

  getUsageStats(): UsageStats {
    return this.usageTracker.getStats();
  }
}
```

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [Swarm Patterns](./SWARM_PATTERNS.md)

---

Built with ❤️ by the Neural Trader team
