/**
 * Competitive pricing analysis with OpenRouter integration
 */

import OpenAI from 'openai';
import { CompetitorAnalysis } from './types';

export class CompetitiveAnalyzer {
  private openai: OpenAI;
  private priceHistory: Map<string, number[]>;

  constructor(apiKey?: string) {
    this.openai = new OpenAI({
      apiKey: apiKey || process.env.OPENROUTER_API_KEY,
      baseURL: 'https://openrouter.ai/api/v1',
    });
    this.priceHistory = new Map();
  }

  /**
   * Analyze competitor prices
   */
  analyze(competitorPrices: number[]): CompetitorAnalysis {
    if (competitorPrices.length === 0) {
      return {
        prices: [],
        avgPrice: 0,
        minPrice: 0,
        maxPrice: 0,
        priceDispersion: 0,
        marketPosition: 'leader',
        recommendedPosition: 'Set your own price based on value',
      };
    }

    const avgPrice = competitorPrices.reduce((a, b) => a + b, 0) / competitorPrices.length;
    const minPrice = Math.min(...competitorPrices);
    const maxPrice = Math.max(...competitorPrices);

    // Calculate price dispersion (coefficient of variation)
    const variance = competitorPrices.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / competitorPrices.length;
    const stdDev = Math.sqrt(variance);
    const priceDispersion = stdDev / avgPrice;

    // Determine market structure
    let marketPosition: CompetitorAnalysis['marketPosition'];
    let recommendedPosition: string;

    if (priceDispersion < 0.1) {
      // Low dispersion: commoditized market
      marketPosition = 'follower';
      recommendedPosition = 'Match market price or differentiate on non-price factors';
    } else if (priceDispersion < 0.2) {
      // Moderate dispersion: competitive market
      marketPosition = 'follower';
      recommendedPosition = 'Slight undercut (2-5%) to gain market share';
    } else {
      // High dispersion: differentiated market
      marketPosition = 'leader';
      recommendedPosition = 'Price based on value differentiation';
    }

    return {
      prices: competitorPrices,
      avgPrice,
      minPrice,
      maxPrice,
      priceDispersion,
      marketPosition,
      recommendedPosition,
    };
  }

  /**
   * Use LLM for strategic pricing advice
   */
  async getStrategicAdvice(
    myPrice: number,
    competitorPrices: number[],
    marketContext: string
  ): Promise<string> {
    try {
      const analysis = this.analyze(competitorPrices);

      const prompt = `You are a pricing strategy expert. Analyze this competitive situation:

My current price: $${myPrice.toFixed(2)}
Competitor prices: ${competitorPrices.map(p => `$${p.toFixed(2)}`).join(', ')}
Market average: $${analysis.avgPrice.toFixed(2)}
Price dispersion: ${(analysis.priceDispersion * 100).toFixed(1)}%
Market context: ${marketContext}

Provide strategic pricing advice considering:
1. Competitive positioning
2. Price elasticity implications
3. Market structure (commoditized vs differentiated)
4. Revenue optimization opportunities
5. Customer perception

Keep response concise (3-4 sentences).`;

      const response = await this.openai.chat.completions.create({
        model: 'anthropic/claude-3.5-sonnet',
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 200,
      });

      return response.choices[0]?.message?.content || analysis.recommendedPosition;
    } catch (error) {
      console.warn('OpenRouter API error:', error);
      return this.analyze(competitorPrices).recommendedPosition;
    }
  }

  /**
   * Predict competitor response to price change
   */
  predictCompetitorResponse(
    myNewPrice: number,
    currentCompetitorPrices: number[]
  ): { willMatch: boolean; expectedPrices: number[]; confidence: number } {
    if (currentCompetitorPrices.length === 0) {
      return {
        willMatch: false,
        expectedPrices: [],
        confidence: 0,
      };
    }

    const analysis = this.analyze(currentCompetitorPrices);
    const avgPrice = analysis.avgPrice;
    const priceGap = myNewPrice - avgPrice;
    const relativeGap = Math.abs(priceGap) / avgPrice;

    // High dispersion = less likely to match
    // Low dispersion = more likely to match
    const matchProbability = Math.max(0, 1 - analysis.priceDispersion * 5);

    let willMatch = false;
    let expectedPrices = [...currentCompetitorPrices];

    // If we significantly undercut and it's a commoditized market
    if (relativeGap < -0.05 && analysis.priceDispersion < 0.15) {
      willMatch = Math.random() < matchProbability;

      if (willMatch) {
        // Competitors will lower prices by 50% of the gap
        const adjustment = priceGap * 0.5;
        expectedPrices = currentCompetitorPrices.map(p => Math.max(p + adjustment, myNewPrice * 0.95));
      }
    }

    return {
      willMatch,
      expectedPrices,
      confidence: matchProbability,
    };
  }

  /**
   * Track competitor price changes over time
   */
  trackCompetitor(competitorId: string, price: number): void {
    if (!this.priceHistory.has(competitorId)) {
      this.priceHistory.set(competitorId, []);
    }

    const history = this.priceHistory.get(competitorId)!;
    history.push(price);

    // Keep last 100 observations
    if (history.length > 100) {
      history.shift();
    }
  }

  /**
   * Analyze competitor pricing behavior
   */
  getCompetitorBehavior(competitorId: string): {
    avgPrice: number;
    volatility: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  } {
    const history = this.priceHistory.get(competitorId);

    if (!history || history.length < 3) {
      return {
        avgPrice: 0,
        volatility: 0,
        trend: 'stable',
      };
    }

    const avgPrice = history.reduce((a, b) => a + b, 0) / history.length;

    // Calculate volatility (std dev)
    const variance = history.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / history.length;
    const volatility = Math.sqrt(variance) / avgPrice;

    // Determine trend using linear regression
    const n = history.length;
    const xAvg = (n - 1) / 2;
    const yAvg = avgPrice;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
      numerator += (i - xAvg) * (history[i] - yAvg);
      denominator += Math.pow(i - xAvg, 2);
    }

    const slope = numerator / denominator;
    const trend = slope > 0.01 ? 'increasing' : slope < -0.01 ? 'decreasing' : 'stable';

    return {
      avgPrice,
      volatility,
      trend,
    };
  }

  /**
   * Find pricing opportunities (gaps in market)
   */
  findPricingGaps(competitorPrices: number[]): Array<{ lower: number; upper: number; size: number }> {
    if (competitorPrices.length < 2) return [];

    const sorted = [...competitorPrices].sort((a, b) => a - b);
    const gaps: Array<{ lower: number; upper: number; size: number }> = [];

    for (let i = 0; i < sorted.length - 1; i++) {
      const gap = sorted[i + 1] - sorted[i];
      const relativeGap = gap / sorted[i];

      // Significant gap if > 10%
      if (relativeGap > 0.1) {
        gaps.push({
          lower: sorted[i],
          upper: sorted[i + 1],
          size: gap,
        });
      }
    }

    return gaps.sort((a, b) => b.size - a.size);
  }
}
