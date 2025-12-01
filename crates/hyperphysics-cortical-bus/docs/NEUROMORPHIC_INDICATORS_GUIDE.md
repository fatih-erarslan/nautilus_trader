# Neuromorphic Technical Indicators Guide

## Using pBit Architecture for Financial Data Analysis

This guide explains how to use the HyperPhysics neuromorphic cortical bus to create
technical indicators that learn patterns through collective dynamics rather than
calculating fixed formulas.

---

## Table of Contents

1. [Conceptual Foundation](#1-conceptual-foundation)
2. [Architecture Overview](#2-architecture-overview)
3. [Step-by-Step Implementation](#3-step-by-step-implementation)
4. [Coupling Design Patterns](#4-coupling-design-patterns)
5. [Temperature Tuning](#5-temperature-tuning)
6. [Advanced Techniques](#6-advanced-techniques)
7. [Comparison with Traditional Indicators](#7-comparison-with-traditional-indicators)

---

## 1. Conceptual Foundation

### Traditional Indicators vs Neuromorphic Indicators

| Aspect | Traditional (RSI, MACD) | Neuromorphic (pBit) |
|--------|-------------------------|---------------------|
| **Computation** | Fixed formula | Emergent dynamics |
| **Adaptation** | None | Self-organizing |
| **Noise** | Filtered out | Used as feature |
| **Parameters** | Manual tuning | Temperature-controlled |
| **Pattern Learning** | Not possible | Natural |

### The Core Insight

Traditional indicators compute a **deterministic function** of price/volume:

```
RSI = 100 - (100 / (1 + RS))
where RS = average_gain / average_loss
```

Neuromorphic indicators let the signal **emerge** from collective behavior:

```
Signal = f(collective_pBit_states)
where states evolve via Metropolis dynamics
```

### Why This Works

1. **Energy Minimization**: pBits naturally find low-energy configurations that represent "stable" patterns
2. **Stochastic Exploration**: Temperature controls exploration of pattern space
3. **Coupling = Knowledge**: The Jᵢⱼ couplings encode learned correlations
4. **Sparse Activation**: Only relevant pBits activate (like sparse coding in brain)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEUROMORPHIC INDICATOR ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   MARKET DATA                 pBIT FABRIC                    OUTPUT         │
│  ┌──────────┐            ┌─────────────────┐            ┌──────────────┐   │
│  │  OHLCV   │            │  Region 0-63:   │            │              │   │
│  │  Bars    │───Encode──►│  MOMENTUM       │            │   Signal     │   │
│  └──────────┘            ├─────────────────┤            │   [-1, +1]   │   │
│                          │  Region 64-127: │───Decode──►│              │   │
│  ┌──────────┐            │  VOLUME         │            │   Direction  │   │
│  │ Volume   │───Encode──►├─────────────────┤            │   {Bull,Bear}│   │
│  │  Data    │            │  Region 128-191:│            │              │   │
│  └──────────┘            │  VOLATILITY     │            │   Regime     │   │
│                          ├─────────────────┤            │   {Hi,Lo,Norm}│  │
│  ┌──────────┐            │  Region 192-255:│            └──────────────┘   │
│  │ Derived  │───Encode──►│  TREND          │                              │
│  │ Features │            └────────┬────────┘                              │
│  └──────────┘                     │                                        │
│                                   ▼                                        │
│                          ┌─────────────────┐                              │
│                          │  METROPOLIS     │                              │
│                          │  DYNAMICS       │                              │
│                          │  (10 sweeps)    │                              │
│                          └─────────────────┘                              │
│                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Three Stages

#### Stage 1: Encoding (Market → Spikes)

Convert market data into pBit bias modulations:

```rust
// Example: Price momentum encoding
let momentum = (current_price - price_n_bars_ago) / price_n_bars_ago;

// Map to pBit index (0-63 for momentum region)
let level = ((momentum / threshold + 4.0).clamp(0.0, 7.0)) as usize;
let pbit_idx = period_index * 8 + level;

// Set bias (strength of input)
fabric.set_bias(pbit_idx, momentum.abs().min(1.0) as f32);
```

#### Stage 2: Dynamics (Metropolis Sweeps)

Let the pBit fabric evolve:

```rust
// Run multiple sweeps to let patterns emerge
for _ in 0..sweeps_per_bar {
    fabric.metropolis_sweep();
}
```

The Metropolis algorithm:
1. For each pBit i, calculate ΔE if we flip it
2. If ΔE ≤ 0: flip (energy decreases = good)
3. If ΔE > 0: flip with probability exp(-ΔE/T)

#### Stage 3: Decoding (States → Signal)

Read the collective state to generate signal:

```rust
// Weight contributions from different regions
let momentum_score = count_active(0..64) / 64.0;
let volume_score = count_active(64..128) / 64.0;
let trend_score = count_active(192..256) / 64.0;

let signal = 0.4 * momentum_score + 0.2 * volume_score + 0.4 * trend_score;
```

---

## 3. Step-by-Step Implementation

### Step 1: Define Your pBit Regions

Decide what each region of pBits will represent:

```rust
// 256 pBits divided into 4 regions of 64 each
const MOMENTUM_REGION: Range<usize> = 0..64;
const VOLUME_REGION: Range<usize> = 64..128;
const VOLATILITY_REGION: Range<usize> = 128..192;
const TREND_REGION: Range<usize> = 192..256;
```

### Step 2: Create the pBit Fabric

```rust
use hyperphysics_cortical_bus::scalable_pbit::{
    ScalablePBitFabric, ScalablePBitConfig
};

let config = ScalablePBitConfig {
    num_pbits: 256,
    avg_degree: 12,      // Average connections per pBit
    temperature: 1.0,    // Start at "warm" temperature
    use_simd: true,
    coupling_capacity: Some(256 * 12),
};

let mut fabric = ScalablePBitFabric::new(config);
```

### Step 3: Design Couplings (The "Knowledge")

This is where domain knowledge is encoded:

```rust
// INTRA-REGION: Nearby pBits in same region should correlate
for i in 0..64 {
    for j in (i+1)..(i+4).min(64) {
        fabric.add_coupling(i, j, 0.5);  // Positive = attract
    }
}

// CROSS-REGION: Encode known relationships
// Example: High volume + high momentum = strong trend
for vol_pbit in 96..128 {  // High volume region
    for trend_pbit in 224..256 {  // Strong trend region
        fabric.add_coupling(vol_pbit, trend_pbit, 0.3);
    }
}

// ANTI-CORRELATION: Encode divergences
// Example: Short-term vs long-term momentum divergence
for short in 0..8 {
    for long in 48..56 {
        fabric.add_coupling(short, long, -0.4);  // Negative = repel
    }
}
```

### Step 4: Encode Market Data

```rust
fn encode_bar(&mut self, bar: &Bar) {
    // Clear previous biases
    for i in 0..256 {
        self.fabric.set_bias(i, 0.0);
    }
    
    // Encode momentum (multiple timeframes)
    for (period_idx, period) in [1, 5, 10, 20].iter().enumerate() {
        let momentum = self.calculate_momentum(*period);
        let level = self.quantize_momentum(momentum);
        let pbit_idx = period_idx * 8 + level;
        self.fabric.set_bias(pbit_idx, momentum.abs() as f32);
    }
    
    // Encode volume
    let volume_ratio = bar.volume / self.avg_volume;
    let vol_level = (volume_ratio.clamp(0.5, 3.0) * 8.0) as usize;
    self.fabric.set_bias(64 + vol_level, volume_ratio as f32);
    
    // ... encode other features
}
```

### Step 5: Run Dynamics and Decode

```rust
fn update(&mut self, bar: &Bar) -> f64 {
    // Encode
    self.encode_bar(bar);
    
    // Dynamics
    for _ in 0..10 {
        self.fabric.metropolis_sweep();
    }
    
    // Decode
    let signal = self.decode_signal();
    
    // Smooth (optional)
    self.smoothed = 0.2 * signal + 0.8 * self.smoothed;
    
    self.smoothed
}

fn decode_signal(&self) -> f64 {
    let momentum_active = (0..64)
        .filter(|&i| self.fabric.get_state(i))
        .map(|i| if i % 8 > 4 { 1.0 } else { -1.0 })
        .sum::<f64>() / 64.0;
    
    let trend_active = (192..256)
        .filter(|&i| self.fabric.get_state(i))
        .count() as f64 / 64.0;
    
    0.5 * momentum_active + 0.5 * (trend_active * 2.0 - 1.0)
}
```

---

## 4. Coupling Design Patterns

### Pattern 1: Momentum Cascade

```
Short-term (1-bar) ──► Medium-term (10-bar) ──► Long-term (50-bar)
      J = 0.5              J = 0.3                J = 0.2
```

Trends should propagate from short to long term:

```rust
// Short-term momentum should eventually align with long-term
fabric.add_coupling(short_term_pbit, medium_term_pbit, 0.5);
fabric.add_coupling(medium_term_pbit, long_term_pbit, 0.3);
```

### Pattern 2: Volume-Volatility Confirmation

```
High Volume ◄───J = 0.4───► High Volatility
     │                           │
     └─────────J = 0.3───────────┘
              (both confirm move)
```

```rust
for vol in high_volume_region {
    for volatility in high_volatility_region {
        fabric.add_coupling(vol, volatility, 0.4);
    }
}
```

### Pattern 3: Divergence Detection

```
Rising Price     ◄───J = -0.5───►    Falling Volume
(bullish signal)                     (bearish warning)
```

Negative coupling creates "tension" when both activate:

```rust
for bullish in bullish_momentum_region {
    for low_vol in low_volume_region {
        fabric.add_coupling(bullish, low_vol, -0.5);  // Divergence!
    }
}
```

### Pattern 4: Regime Clustering

```
Low Volatility cluster ◄──► Trending cluster
(all low-vol pBits connected)

High Volatility cluster ◄──► Mean-reversion cluster
```

```rust
// Low-vol pBits cluster together
for i in low_vol_region {
    for j in (i+1)..low_vol_region.end {
        fabric.add_coupling(i, j, 0.6);
    }
}
```

---

## 5. Temperature Tuning

Temperature T controls the trade-off between:
- **Exploration** (high T): Random, noisy signals
- **Exploitation** (low T): Stable, confident signals

### Adaptive Temperature

```rust
fn adaptive_temperature(&mut self, volatility: f64) -> f64 {
    // High market volatility → high pBit temperature
    // (more exploration when market is uncertain)
    let base_temp = 1.0;
    let vol_factor = volatility / self.avg_volatility;
    base_temp * vol_factor.clamp(0.5, 2.0)
}
```

### Annealing Schedule

For one-shot pattern detection:

```rust
fn annealed_signal(&mut self, bar: &Bar) -> f64 {
    self.encode_bar(bar);
    
    // Start hot (exploration)
    for temp in [5.0, 2.0, 1.0, 0.5, 0.2] {
        self.fabric.set_temperature(temp);
        for _ in 0..5 {
            self.fabric.metropolis_sweep();
        }
    }
    
    // Final state is the "answer"
    self.decode_signal()
}
```

---

## 6. Advanced Techniques

### 6.1 Multi-Timeframe Indicator

Run parallel fabrics at different timeframes:

```rust
struct MultiTimeframeIndicator {
    short_term: NeuromorphicIndicator,  // 1-min bars
    medium_term: NeuromorphicIndicator, // 5-min bars
    long_term: NeuromorphicIndicator,   // 1-hour bars
}

impl MultiTimeframeIndicator {
    fn combined_signal(&self) -> f64 {
        0.5 * self.short_term.signal()
        + 0.3 * self.medium_term.signal()
        + 0.2 * self.long_term.signal()
    }
}
```

### 6.2 Learning Couplings from Data

Instead of hand-designing couplings, learn them:

```rust
fn learn_couplings(&mut self, historical_bars: &[Bar], labels: &[f64]) {
    // For each pair of pBit regions, calculate correlation
    // of activations with future returns
    
    for bar in historical_bars {
        self.encode_bar(bar);
        self.run_dynamics();
        
        // Record which pBits activated together
        for i in 0..256 {
            for j in (i+1)..256 {
                if self.fabric.get_state(i) && self.fabric.get_state(j) {
                    self.co_activation[i][j] += 1;
                }
            }
        }
    }
    
    // Convert co-activations to couplings weighted by label correlation
    // (Simplified Hebbian learning: "neurons that fire together wire together")
}
```

### 6.3 Spike-Based Streaming

For real-time data with sub-millisecond updates:

```rust
fn process_tick(&mut self, tick: &Tick) {
    // Generate spike from tick
    let spike = self.encode_tick(tick);
    
    // Inject spike into running fabric (no reset)
    self.fabric.set_bias(spike.pbit_idx, spike.strength);
    
    // Single sweep (not multiple)
    self.fabric.metropolis_sweep();
    
    // Check for output spike (state change)
    if self.state_changed() {
        emit_signal(self.decode_signal());
    }
}
```

---

## 7. Comparison with Traditional Indicators

### Example: Momentum Indicator

**Traditional RSI:**
```python
def RSI(prices, period=14):
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = moving_average(gains, period)
    avg_loss = moving_average(losses, period)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
```

Fixed formula, fixed period, no adaptation.

**Neuromorphic Momentum:**
```rust
// pBit momentum indicator
// - Encodes multiple timeframes simultaneously
// - Learns correlations between timeframes
// - Adapts to volatility regime
// - Output emerges from collective dynamics

fn neuromorphic_momentum(&mut self, bar: &Bar) -> f64 {
    self.encode_momentum_features(bar);  // Multi-timeframe
    self.run_dynamics(10);                // Let patterns emerge
    self.decode_momentum_signal()         // Collective answer
}
```

### When to Use Neuromorphic

| Use Case | Traditional Better | Neuromorphic Better |
|----------|-------------------|---------------------|
| Simple trend following | ✅ | |
| Mean reversion | ✅ | |
| Complex pattern detection | | ✅ |
| Regime adaptation | | ✅ |
| Multi-factor analysis | | ✅ |
| Noisy data | | ✅ |
| Real-time streaming | | ✅ |

---

## Quick Start

```bash
# Run the demo
cargo run --example financial_indicators --release --no-default-features
```

This will:
1. Generate synthetic market data
2. Process it through a neuromorphic indicator
3. Show signals and regime detection
4. Visualize the output

---

## Summary

1. **Encode** market data into pBit biases (different regions for different features)
2. **Design couplings** that encode your market knowledge
3. **Run dynamics** to let patterns emerge
4. **Decode** the collective state into a signal
5. **Tune temperature** for exploration/exploitation trade-off

The key insight: **You're not calculating a formula. You're letting the answer emerge from physics-inspired collective dynamics.**
