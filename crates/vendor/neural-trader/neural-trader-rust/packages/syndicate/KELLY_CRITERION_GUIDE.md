# Kelly Criterion Guide

A comprehensive guide to understanding and implementing the Kelly Criterion for optimal bet sizing.

## Table of Contents

- [Introduction](#introduction)
- [The Formula](#the-formula)
- [Mathematical Foundation](#mathematical-foundation)
- [Fractional Kelly](#fractional-kelly)
- [Implementation](#implementation)
- [Examples](#examples)
- [Edge Cases](#edge-cases)
- [Best Practices](#best-practices)
- [Common Mistakes](#common-mistakes)
- [Advanced Topics](#advanced-topics)

## Introduction

The Kelly Criterion is a mathematical formula developed by John L. Kelly Jr. in 1956 for optimal capital allocation. It determines the theoretically optimal bet size that maximizes the expected logarithm of wealth over time.

### Key Benefits

1. **Optimal Growth**: Maximizes long-term geometric growth rate
2. **Risk Management**: Never risks full bankroll on single bet
3. **Mathematical Proof**: Backed by rigorous mathematical proof
4. **Adaptability**: Adjusts bet size based on edge and odds
5. **Capital Preservation**: Protects against ruin

### When to Use Kelly

✅ **Good for:**
- Sports betting with known edge
- Trading with statistical advantage
- Long-term wealth maximization
- Professional gambling
- Portfolio allocation

❌ **Not ideal for:**
- Recreational betting
- Unknown probabilities
- Highly correlated bets
- Short-term goals
- Risk-averse investors

## The Formula

### Basic Formula

```
f* = (bp - q) / b
```

Where:
- `f*` = fraction of bankroll to bet (Kelly percentage)
- `b` = decimal odds - 1 (net odds)
- `p` = probability of winning
- `q` = probability of losing (1 - p)

### Alternative Formula (American Odds)

```
f* = (p * odds - 1) / odds
```

For American odds, convert to decimal first:
- Positive odds: `decimal = (american / 100) + 1`
- Negative odds: `decimal = (100 / |american|) + 1`

### Edge Formula

The edge is the expected profit per dollar bet:

```
edge = (p * b) - q = bp - q
```

Kelly simplifies to:
```
f* = edge / b
```

## Mathematical Foundation

### Expected Value

The Kelly Criterion maximizes the expected logarithm of wealth:

```
E[log(W)] = p * log(1 + b*f) + q * log(1 - f)
```

Taking the derivative and setting to zero:
```
p * b / (1 + b*f) - q / (1 - f) = 0
```

Solving for f gives the Kelly formula.

### Growth Rate

The geometric growth rate with Kelly betting is:

```
G = p * log(1 + b*f*) + q * log(1 - f*)
```

This growth rate is maximized when using the Kelly percentage.

### Proof of Optimality

Kelly's strategy is optimal because:

1. It maximizes the expected log utility
2. It achieves the highest geometric growth rate
3. It reaches any wealth goal in minimum expected time
4. It minimizes the expected time to reach a goal

## Fractional Kelly

Full Kelly can be aggressive and lead to high volatility. Fractional Kelly reduces bet size while maintaining good growth:

### Comparison Table

| Fraction | Growth Rate | Volatility | Max Drawdown | Use Case |
|----------|-------------|------------|--------------|----------|
| Full (1.0) | 100% | Very High | ~40% | Maximum growth |
| 3/4 (0.75) | 94% | High | ~30% | Aggressive |
| Half (0.5) | 75% | Medium | ~20% | Balanced |
| 1/3 (0.33) | 60% | Medium-Low | ~15% | Conservative |
| Quarter (0.25) | 50% | Low | ~10% | Very Conservative |
| 1/8 (0.125) | 25% | Very Low | ~5% | Ultra Conservative |

### Formula

```
Fractional Kelly = Full Kelly * Fraction
```

Example:
```
Full Kelly = 14.1%
Quarter Kelly = 14.1% * 0.25 = 3.525%
```

### Recommended Fractions

Based on risk tolerance:

```typescript
const fractions = {
  aggressive: 0.50,      // Half Kelly
  moderate: 0.33,        // Third Kelly
  conservative: 0.25,    // Quarter Kelly (recommended)
  veryConservative: 0.125 // Eighth Kelly
};
```

## Implementation

### TypeScript Implementation

```typescript
interface KellyOptions {
  odds: number;          // Decimal odds
  probability: number;   // Win probability (0-1)
  bankroll: number;      // Current bankroll
  fractionalKelly?: number; // Fractional Kelly (default: 0.25)
}

interface KellyResult {
  kellyPercentage: number;      // Full Kelly percentage
  fractionalPercentage: number; // Adjusted Kelly percentage
  betSize: number;              // Bet amount in dollars
  expectedValue: number;        // Expected profit
  edge: number;                 // Edge percentage
  riskOfRuin: number;          // Risk of ruin percentage
}

class KellyCalculator {
  calculate(options: KellyOptions): KellyResult {
    const {
      odds,
      probability,
      bankroll,
      fractionalKelly = 0.25
    } = options;

    // Validate inputs
    this.validate(odds, probability, bankroll);

    // Calculate Kelly percentage
    const b = odds - 1;  // Net odds
    const p = probability;
    const q = 1 - p;

    const kelly = (b * p - q) / b;

    // Handle negative Kelly (no edge)
    if (kelly <= 0) {
      return {
        kellyPercentage: 0,
        fractionalPercentage: 0,
        betSize: 0,
        expectedValue: 0,
        edge: b * p - q,
        riskOfRuin: 1.0
      };
    }

    // Apply fractional Kelly
    const adjustedKelly = kelly * fractionalKelly;

    // Calculate bet size
    const betSize = bankroll * adjustedKelly;

    // Calculate expected value
    const expectedValue = betSize * (b * p - q);

    // Calculate edge
    const edge = b * p - q;

    // Calculate risk of ruin (approximation)
    const riskOfRuin = Math.exp(-2 * edge * bankroll / betSize);

    return {
      kellyPercentage: kelly,
      fractionalPercentage: adjustedKelly,
      betSize,
      expectedValue,
      edge,
      riskOfRuin
    };
  }

  private validate(odds: number, probability: number, bankroll: number): void {
    if (odds <= 1) {
      throw new Error('Odds must be greater than 1');
    }
    if (probability <= 0 || probability >= 1) {
      throw new Error('Probability must be between 0 and 1');
    }
    if (bankroll <= 0) {
      throw new Error('Bankroll must be positive');
    }
  }

  calculateEdge(odds: number, probability: number): number {
    const b = odds - 1;
    const p = probability;
    const q = 1 - p;
    return b * p - q;
  }

  hasEdge(odds: number, probability: number): boolean {
    return this.calculateEdge(odds, probability) > 0;
  }
}
```

### Usage Examples

```typescript
const calculator = new KellyCalculator();

// Example 1: NFL game
const result1 = calculator.calculate({
  odds: 2.10,           // Chiefs at 2.10
  probability: 0.55,    // 55% chance
  bankroll: 100000,     // $100k bankroll
  fractionalKelly: 0.25 // Quarter Kelly
});

console.log(`Full Kelly: ${result1.kellyPercentage.toFixed(3)} (${(result1.kellyPercentage * 100).toFixed(1)}%)`);
console.log(`Quarter Kelly: ${result1.fractionalPercentage.toFixed(3)} (${(result1.fractionalPercentage * 100).toFixed(1)}%)`);
console.log(`Bet Size: $${result1.betSize.toFixed(2)}`);
console.log(`Expected Value: $${result1.expectedValue.toFixed(2)}`);
console.log(`Edge: ${(result1.edge * 100).toFixed(2)}%`);

// Example 2: High edge opportunity
const result2 = calculator.calculate({
  odds: 3.00,           // Underdog at 3.00
  probability: 0.40,    // 40% chance (good value)
  bankroll: 100000,
  fractionalKelly: 0.25
});

// Example 3: Low edge
const result3 = calculator.calculate({
  odds: 1.95,           // Even money
  probability: 0.52,    // Small edge
  bankroll: 100000,
  fractionalKelly: 0.25
});
```

## Examples

### Example 1: NFL Moneyline

**Scenario:**
- Kansas City Chiefs vs Buffalo Bills
- Market odds: Chiefs 2.10 (implied 47.6%)
- Your probability: 55%
- Bankroll: $100,000
- Strategy: Quarter Kelly

**Calculation:**
```typescript
const odds = 2.10;
const probability = 0.55;
const bankroll = 100000;

// Calculate Kelly
const b = odds - 1;  // 1.10
const p = probability;  // 0.55
const q = 1 - p;  // 0.45

const kelly = (b * p - q) / b;
// kelly = (1.10 * 0.55 - 0.45) / 1.10
// kelly = (0.605 - 0.45) / 1.10
// kelly = 0.155 / 1.10
// kelly = 0.141 (14.1%)

// Apply Quarter Kelly
const fractionalKelly = kelly * 0.25;  // 0.03525 (3.525%)

// Calculate bet size
const betSize = bankroll * fractionalKelly;  // $3,525

// Expected value
const edge = b * p - q;  // 0.155 (15.5% edge)
const expectedValue = betSize * edge;  // $546.38
```

**Result:**
- Full Kelly: 14.1% of bankroll
- Quarter Kelly: 3.525% of bankroll
- Bet size: $3,525
- Expected profit: $546.38
- Edge: 15.5%

### Example 2: NHL Spread

**Scenario:**
- Toronto Maple Leafs -1.5
- Market odds: 2.80
- Your probability: 45%
- Bankroll: $50,000
- Strategy: Half Kelly

**Calculation:**
```typescript
const odds = 2.80;
const probability = 0.45;
const bankroll = 50000;

const b = 1.80;
const p = 0.45;
const q = 0.55;

const kelly = (1.80 * 0.45 - 0.55) / 1.80;
// kelly = (0.81 - 0.55) / 1.80
// kelly = 0.26 / 1.80
// kelly = 0.144 (14.4%)

const halfKelly = kelly * 0.5;  // 0.072 (7.2%)
const betSize = bankroll * halfKelly;  // $3,600

const edge = b * p - q;  // 0.26 (26% edge)
const expectedValue = betSize * edge;  // $936
```

**Result:**
- Full Kelly: 14.4%
- Half Kelly: 7.2%
- Bet size: $3,600
- Expected profit: $936
- Edge: 26%

### Example 3: NBA Over/Under

**Scenario:**
- Lakers vs Celtics Over 225.5
- Market odds: 1.91
- Your probability: 53%
- Bankroll: $200,000
- Strategy: Third Kelly

**Calculation:**
```typescript
const odds = 1.91;
const probability = 0.53;
const bankroll = 200000;

const b = 0.91;
const p = 0.53;
const q = 0.47;

const kelly = (0.91 * 0.53 - 0.47) / 0.91;
// kelly = (0.4823 - 0.47) / 0.91
// kelly = 0.0123 / 0.91
// kelly = 0.0135 (1.35%)

const thirdKelly = kelly * 0.33;  // 0.00446 (0.446%)
const betSize = bankroll * thirdKelly;  // $891

const edge = b * p - q;  // 0.0123 (1.23% edge)
const expectedValue = betSize * edge;  // $10.96
```

**Result:**
- Full Kelly: 1.35%
- Third Kelly: 0.446%
- Bet size: $891
- Expected profit: $10.96
- Edge: 1.23%

### Example 4: No Edge Scenario

**Scenario:**
- Fair coin flip
- Market odds: 2.00
- Your probability: 50%
- Bankroll: $100,000

**Calculation:**
```typescript
const odds = 2.00;
const probability = 0.50;

const b = 1.00;
const p = 0.50;
const q = 0.50;

const kelly = (1.00 * 0.50 - 0.50) / 1.00;
// kelly = (0.50 - 0.50) / 1.00
// kelly = 0.00 / 1.00
// kelly = 0 (0%)

// No bet should be made!
```

**Result:**
- Full Kelly: 0%
- Don't bet!
- Edge: 0%

## Edge Cases

### Negative Kelly

When Kelly is negative, you have no edge and should not bet:

```typescript
if (kelly <= 0) {
  console.log('No edge - do not bet');
  return { betSize: 0 };
}
```

### Kelly > 1

When Kelly exceeds 100%, you have a massive edge:

```typescript
if (kelly > 1) {
  // This is rare but possible with huge edges
  console.log(`Massive edge: ${(kelly * 100).toFixed(1)}%`);
  // Consider still capping at some reasonable fraction
  const cappedKelly = Math.min(kelly, 0.50);
}
```

### Very Small Kelly

When Kelly is very small (<0.5%), consider not betting:

```typescript
if (kelly < 0.005) {
  console.log('Edge too small - skip bet');
  return { betSize: 0 };
}
```

### Uncertain Probabilities

When probability estimates are uncertain, reduce Kelly:

```typescript
const confidenceAdjustment = {
  high: 1.0,      // 90%+ confidence
  medium: 0.75,   // 70-90% confidence
  low: 0.50,      // 50-70% confidence
  veryLow: 0.25   // <50% confidence
};

const adjustedKelly = kelly * confidenceAdjustment.medium;
```

## Best Practices

### 1. Conservative Fractions

Use Quarter Kelly or less for safety:

```typescript
const recommendedFractions = {
  professional: 0.25,      // Quarter Kelly
  intermediate: 0.167,     // 1/6 Kelly
  beginner: 0.125         // Eighth Kelly
};
```

### 2. Probability Accuracy

Kelly is only as good as your probability estimates:

```typescript
// Track your calibration
class ProbabilityTracker {
  predictions: Array<{ probability: number; won: boolean }> = [];

  add(probability: number, won: boolean) {
    this.predictions.push({ probability, won });
  }

  getCalibration(): number {
    // Compare predicted vs actual win rates
    const buckets = [0.5, 0.6, 0.7, 0.8, 0.9];
    // Calculate calibration error
  }
}
```

### 3. Bankroll Management

Update bankroll regularly:

```typescript
class BankrollTracker {
  private currentBankroll: number;

  updateBankroll(amount: number) {
    this.currentBankroll = amount;
  }

  recalculateKelly(odds: number, probability: number) {
    // Always use current bankroll for Kelly calculation
    return calculator.calculate({
      odds,
      probability,
      bankroll: this.currentBankroll,
      fractionalKelly: 0.25
    });
  }
}
```

### 4. Correlation Awareness

Reduce Kelly for correlated bets:

```typescript
const correlationAdjustment = {
  independent: 1.0,
  lowCorrelation: 0.80,
  mediumCorrelation: 0.60,
  highCorrelation: 0.40
};

// If betting multiple correlated games
const adjustedKelly = kelly * correlationAdjustment.mediumCorrelation;
```

### 5. Drawdown Protection

Reduce Kelly during drawdowns:

```typescript
class DrawdownProtection {
  reduceKellyOnDrawdown(
    kelly: number,
    currentDrawdown: number
  ): number {
    if (currentDrawdown > 0.10) {
      return kelly * 0.50;  // Half Kelly at 10% drawdown
    }
    if (currentDrawdown > 0.20) {
      return kelly * 0.25;  // Quarter Kelly at 20% drawdown
    }
    return kelly;
  }
}
```

## Common Mistakes

### 1. Using Full Kelly

**Mistake:**
```typescript
const betSize = bankroll * kelly;  // Too aggressive!
```

**Fix:**
```typescript
const betSize = bankroll * kelly * 0.25;  // Quarter Kelly
```

### 2. Overconfident Probabilities

**Mistake:**
```typescript
const probability = 0.90;  // Overconfident
```

**Fix:**
```typescript
// Adjust probabilities toward market (regression to mean)
const marketProbability = 1 / odds;
const adjustedProbability = 0.7 * myProbability + 0.3 * marketProbability;
```

### 3. Ignoring Correlation

**Mistake:**
```typescript
// Betting full Kelly on multiple correlated games
const betSize1 = bankroll * kelly1;
const betSize2 = bankroll * kelly2;  // Same bankroll!
```

**Fix:**
```typescript
// Reduce for correlation and split bankroll
const availablePerBet = bankroll / numberOfBets;
const betSize1 = availablePerBet * kelly1 * correlationFactor;
```

### 4. Not Updating Bankroll

**Mistake:**
```typescript
// Using stale bankroll
const betSize = originalBankroll * kelly;
```

**Fix:**
```typescript
// Always use current bankroll
const betSize = getCurrentBankroll() * kelly;
```

### 5. Betting Without Edge

**Mistake:**
```typescript
// Betting even with negative Kelly
if (kelly < 0) {
  const betSize = bankroll * Math.abs(kelly);  // Wrong!
}
```

**Fix:**
```typescript
if (kelly <= 0) {
  return { betSize: 0 };  // Don't bet!
}
```

## Advanced Topics

### Simultaneous Kelly

For multiple simultaneous bets:

```typescript
class SimultaneousKelly {
  calculateOptimal(opportunities: Opportunity[]): Allocation[] {
    // Solve system of equations for optimal allocations
    // Considers correlation matrix
    // Maximizes total geometric growth
  }
}
```

### Dynamic Kelly

Adjust Kelly based on market conditions:

```typescript
class DynamicKelly {
  adjustForVolatility(kelly: number, volatility: number): number {
    // Reduce Kelly in high volatility
    return kelly * (1 / (1 + volatility));
  }

  adjustForLiquidity(kelly: number, liquidity: number): number {
    // Reduce Kelly for illiquid markets
    return kelly * Math.min(1, liquidity / targetLiquidity);
  }
}
```

### Kelly with Uncertainty

Account for probability uncertainty:

```typescript
class UncertaintyAdjustedKelly {
  calculate(
    odds: number,
    probabilityMean: number,
    probabilityStd: number,
    fractionalKelly: number
  ): number {
    // Use Bayesian approach
    // Integrate over probability distribution
    // Results in more conservative Kelly
  }
}
```

## References

1. Kelly, J. L. (1956). "A New Interpretation of Information Rate"
2. Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market"
3. MacLean, L. C., et al. (2011). "The Kelly Capital Growth Investment Criterion"

---

**Remember: Kelly is a powerful tool, but use it wisely with fractional sizing and accurate probabilities!**
