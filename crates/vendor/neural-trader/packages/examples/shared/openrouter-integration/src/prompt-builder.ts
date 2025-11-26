/**
 * Prompt builder for structured LLM interactions
 * Provides templates and utilities for consistent prompt engineering
 */

export interface PromptTemplate {
  system: string;
  user: string;
  variables?: Record<string, string>;
}

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export class PromptBuilder {
  private messages: Message[] = [];
  private templates: Map<string, PromptTemplate> = new Map();

  constructor() {
    this.initializeDefaultTemplates();
  }

  /**
   * Add a system message
   */
  system(content: string): this {
    this.messages.push({ role: 'system', content });
    return this;
  }

  /**
   * Add a user message
   */
  user(content: string): this {
    this.messages.push({ role: 'user', content });
    return this;
  }

  /**
   * Add an assistant message
   */
  assistant(content: string): this {
    this.messages.push({ role: 'assistant', content });
    return this;
  }

  /**
   * Use a predefined template
   */
  useTemplate(
    name: string,
    variables?: Record<string, string>
  ): this {
    const template = this.templates.get(name);
    if (!template) {
      throw new Error(`Template "${name}" not found`);
    }

    const mergedVars = { ...template.variables, ...variables };

    this.system(this.interpolate(template.system, mergedVars));
    this.user(this.interpolate(template.user, mergedVars));

    return this;
  }

  /**
   * Register a custom template
   */
  registerTemplate(name: string, template: PromptTemplate): void {
    this.templates.set(name, template);
  }

  /**
   * Build the messages array
   */
  build(): Message[] {
    if (this.messages.length === 0) {
      throw new Error('Cannot build empty prompt');
    }
    return [...this.messages];
  }

  /**
   * Clear all messages
   */
  clear(): this {
    this.messages = [];
    return this;
  }

  /**
   * Create a new builder instance from template
   */
  static fromTemplate(
    name: string,
    variables?: Record<string, string>
  ): PromptBuilder {
    const builder = new PromptBuilder();
    return builder.useTemplate(name, variables);
  }

  /**
   * Initialize default templates
   */
  private initializeDefaultTemplates(): void {
    // Trading Strategy Analysis
    this.templates.set('trading-analysis', {
      system: `You are an expert quantitative trading analyst specializing in {strategy_type} strategies.
Your role is to analyze market data, identify patterns, and provide actionable trading insights.
Focus on risk management, statistical significance, and practical implementation.`,
      user: `Analyze the following market scenario:

Market Data: {market_data}
Timeframe: {timeframe}
Strategy: {strategy_type}

Provide:
1. Key patterns and signals
2. Risk assessment (probability of success, max drawdown)
3. Entry/exit points with confidence levels
4. Position sizing recommendations`,
      variables: {
        strategy_type: 'momentum',
        timeframe: '1-day',
        market_data: 'SPY, QQQ daily prices',
      },
    });

    // Code Generation
    this.templates.set('code-generation', {
      system: `You are an expert software engineer specializing in {language}.
Write clean, efficient, well-documented code following best practices and design patterns.
Include error handling, type safety, and comprehensive comments.`,
      user: `Generate {language} code for the following requirement:

Task: {task}
Constraints: {constraints}
Performance Requirements: {performance}

Provide:
1. Complete, production-ready code
2. Time and space complexity analysis
3. Test cases covering edge cases
4. Usage examples`,
      variables: {
        language: 'TypeScript',
        task: 'Data processing function',
        constraints: 'None',
        performance: 'O(n) time complexity',
      },
    });

    // Data Analysis
    this.templates.set('data-analysis', {
      system: `You are a data scientist specializing in {analysis_type} analysis.
Provide statistical rigor, clear visualizations recommendations, and actionable insights.
Focus on data quality, methodology, and reproducibility.`,
      user: `Analyze the following dataset:

Data Description: {data_description}
Analysis Goal: {goal}
Metrics of Interest: {metrics}

Provide:
1. Exploratory data analysis summary
2. Statistical tests and significance levels
3. Key findings and patterns
4. Visualization recommendations
5. Actionable recommendations`,
      variables: {
        analysis_type: 'time-series',
        data_description: 'Financial market data',
        goal: 'Identify trends and anomalies',
        metrics: 'Mean, variance, correlation',
      },
    });

    // Research and Reasoning
    this.templates.set('research', {
      system: `You are a research assistant with expertise in {domain}.
Provide well-reasoned analysis backed by evidence and logical arguments.
Consider multiple perspectives and acknowledge limitations.`,
      user: `Research the following topic:

Topic: {topic}
Scope: {scope}
Depth: {depth}

Provide:
1. Background and context
2. Current state of knowledge
3. Key insights and findings
4. Open questions and future directions
5. Relevant references and resources`,
      variables: {
        domain: 'quantitative finance',
        topic: 'Machine learning in trading',
        scope: 'Recent developments (2023-2024)',
        depth: 'Comprehensive overview',
      },
    });

    // Decision Making
    this.templates.set('decision', {
      system: `You are a decision analysis expert specializing in {decision_type} decisions.
Use structured frameworks like cost-benefit analysis, decision trees, and risk matrices.
Provide clear recommendations with confidence levels.`,
      user: `Help make a decision about:

Decision: {decision}
Options: {options}
Constraints: {constraints}
Success Criteria: {criteria}

Provide:
1. Option comparison matrix
2. Risk analysis for each option
3. Expected outcomes and probabilities
4. Recommended decision with reasoning
5. Contingency plans`,
      variables: {
        decision_type: 'strategic',
        decision: 'Investment allocation',
        options: 'Multiple portfolio configurations',
        constraints: 'Risk tolerance, time horizon',
        criteria: 'Risk-adjusted returns',
      },
    });

    // Optimization
    this.templates.set('optimization', {
      system: `You are an optimization expert specializing in {optimization_type} optimization.
Apply mathematical rigor, consider constraints, and provide practical solutions.
Focus on global optima while acknowledging computational limitations.`,
      user: `Optimize the following problem:

Problem: {problem}
Objective: {objective}
Constraints: {constraints}
Variables: {variables}

Provide:
1. Problem formulation (mathematical notation)
2. Recommended optimization approach
3. Expected solution quality
4. Implementation guidance
5. Sensitivity analysis`,
      variables: {
        optimization_type: 'portfolio',
        problem: 'Asset allocation',
        objective: 'Maximize Sharpe ratio',
        constraints: 'Budget, risk limits',
        variables: 'Asset weights',
      },
    });
  }

  /**
   * Interpolate variables into template string
   */
  private interpolate(template: string, variables: Record<string, string>): string {
    return template.replace(/{(\w+)}/g, (match, key) => {
      return variables[key] !== undefined ? variables[key] : match;
    });
  }

  /**
   * Get available templates
   */
  getAvailableTemplates(): string[] {
    return Array.from(this.templates.keys());
  }

  /**
   * Get template details
   */
  getTemplate(name: string): PromptTemplate | undefined {
    return this.templates.get(name);
  }

  /**
   * Export current conversation
   */
  export(): Message[] {
    return this.build();
  }

  /**
   * Import conversation from messages
   */
  import(messages: Message[]): this {
    this.messages = [...messages];
    return this;
  }

  /**
   * Count total tokens (rough estimate)
   */
  estimateTokens(): number {
    const text = this.messages.map((m) => m.content).join(' ');
    // Rough estimate: 1 token ≈ 4 characters
    return Math.ceil(text.length / 4);
  }

  /**
   * Create few-shot examples
   */
  addFewShotExamples(examples: Array<{ user: string; assistant: string }>): this {
    examples.forEach((example) => {
      this.user(example.user);
      this.assistant(example.assistant);
    });
    return this;
  }

  /**
   * Add chain-of-thought reasoning prompt
   */
  enableChainOfThought(): this {
    const lastMessage = this.messages[this.messages.length - 1];
    if (lastMessage && lastMessage.role === 'user') {
      lastMessage.content += '\n\nLet\'s think step by step:';
    }
    return this;
  }

  /**
   * Add structured output format
   */
  requestStructuredOutput(format: 'json' | 'markdown' | 'code'): this {
    const formatInstructions = {
      json: 'Respond with valid JSON only. No additional text.',
      markdown: 'Format response as markdown with clear sections.',
      code: 'Provide code only. Include comments but no additional explanation.',
    };

    const lastMessage = this.messages[this.messages.length - 1];
    if (lastMessage && lastMessage.role === 'user') {
      lastMessage.content += `\n\n${formatInstructions[format]}`;
    }
    return this;
  }
}
