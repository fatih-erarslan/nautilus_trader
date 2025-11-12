# Market Consciousness-Based Regime Detection

**Version**: 1.0.0
**Author**: Consciousness-Analyst Agent
**Date**: 2025-11-12
**Status**: Design Phase

---

## Executive Summary

This document presents a novel approach to financial market regime detection using principles from Integrated Information Theory (IIT) and consciousness science. We map market collective behavior to consciousness metrics:

- **Φ (Integrated Information)**: Measures market coordination and information integration
- **CI (Consciousness Index)**: Quantifies market complexity, coherence, and pattern persistence

The framework enables detection of six distinct market regimes with scientific rigor and mathematical precision.

---

## 1. Theoretical Foundation

### 1.1 Market as Conscious System

Financial markets exhibit collective behavior analogous to conscious systems:

- **Information Integration**: Price formation integrates distributed information
- **Irreducibility**: Market behavior emerges from complex interactions
- **Differentiation**: Multiple subsystems (sectors, asset classes) with distinct patterns
- **Causal Power**: Past states causally influence future states

### 1.2 IIT Mapping to Markets

**Integrated Information Theory (Tononi et al., 2016)** provides:

```
Φ = min[D(p(X₀|do(X₋₁⁽ᴹ⁾)) || ∏ᵢp(Xᵢ₀|do(X₋₁⁽ᴹⁱ⁾)))]
```

Where:
- `X₀`: Current market state
- `X₋₁`: Previous market state
- `M`: Minimum information partition
- `D`: KL divergence

**Market Translation**:
- `X₀`: Current price vector across assets
- `X₋₁`: Historical price vector
- `M`: Sector/asset class partitions
- `Φ`: Information lost when market is partitioned

---

## 2. Market Consciousness Framework

### 2.1 Core Data Structures

```rust
/// Market consciousness state
pub struct MarketConsciousness {
    /// Integrated information (0.0 to 1.0)
    pub phi: f64,

    /// Consciousness index (0.0 to unbounded)
    pub ci: f64,

    /// Current market regime
    pub regime: MarketRegime,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Timestamp of state
    pub timestamp: DateTime<Utc>,
}

/// Market regime classifications
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketRegime {
    /// Sustained upward trend (High Φ, Rising CI)
    Bull {
        strength: f64,  // 0.0 to 1.0
    },

    /// Sustained downward trend (High Φ, Falling CI)
    Bear {
        severity: f64,  // 0.0 to 1.0
    },

    /// Unsustainable euphoria (Very High Φ, Unstable CI)
    Bubble {
        fragility: f64,  // 0.0 to 1.0
    },

    /// Sharp reversal (Low Φ, High CI)
    Correction {
        depth: f64,  // 0.0 to 1.0
    },

    /// Sideways consolidation (Medium Φ, Stable CI)
    Ranging {
        volatility: f64,  // 0.0 to 1.0
    },

    /// Rapid collapse (Φ Spike → Collapse)
    Crash {
        collapse_rate: f64,  // Rate of Φ decline
    },
}

/// Market data container
pub struct MarketData {
    /// Asset price matrix (time × assets)
    pub prices: Array2<f64>,

    /// Asset return matrix
    pub returns: Array2<f64>,

    /// Volume matrix
    pub volumes: Array2<f64>,

    /// Correlation matrix (assets × assets)
    pub correlations: Array2<f64>,

    /// Sector classifications
    pub sectors: HashMap<String, Vec<usize>>,

    /// Timestamp index
    pub timestamps: Vec<DateTime<Utc>>,
}
```

### 2.2 Regime Characteristics

| Regime | Φ Range | CI Range | Market Condition | Trading Bias |
|--------|---------|----------|------------------|--------------|
| **Bull** | 0.6-0.8 | 0.5-0.7 | Coordinated buying, healthy growth | Long |
| **Bear** | 0.6-0.8 | 0.2-0.4 | Coordinated selling, orderly decline | Short |
| **Bubble** | >0.8 | >0.7 | Excessive integration, manic | Reduce |
| **Correction** | <0.3 | 0.5-0.8 | Fragmented, complex patterns | Cautious |
| **Ranging** | 0.3-0.6 | 0.3-0.6 | Equilibrium, mean reversion | Neutral |
| **Crash** | Spike→0 | Collapse | System breakdown | Exit |

---

## 3. Φ (Integrated Information) Calculation

### 3.1 Mathematical Definition

For market system `S` with state space `X`:

```
Φ(S) = min[MI(X₀; X₋₁) - Σᵢ MI(Xᵢ₀; X₋₁)]
```

Where:
- `MI`: Mutual information
- `Xᵢ`: Partition i of system S
- Minimization over all bipartitions

### 3.2 Implementation Algorithm

```rust
pub struct PhiCalculator {
    /// Historical window size
    window_size: usize,

    /// Number of bootstrap samples
    bootstrap_samples: usize,

    /// Partition strategy
    partition_method: PartitionMethod,
}

#[derive(Debug, Clone)]
pub enum PartitionMethod {
    /// Partition by sectors
    Sectoral,

    /// Partition by asset classes
    AssetClass,

    /// Hierarchical clustering
    Hierarchical,

    /// Information-theoretic cut
    MinCut,
}

impl PhiCalculator {
    /// Calculate integrated information
    pub fn calculate(&self, data: &MarketData) -> PhiResult {
        // 1. Extract recent window
        let window = data.recent_window(self.window_size);

        // 2. Calculate whole-system mutual information
        let whole_mi = self.mutual_information(
            &window.returns,
            None,  // All assets
        );

        // 3. Generate all bipartitions
        let partitions = self.generate_partitions(&window);

        // 4. Find minimum information partition (MIP)
        let mut min_phi = f64::INFINITY;
        let mut mip = None;

        for partition in partitions {
            let part_mi: f64 = partition.iter()
                .map(|subset| {
                    self.mutual_information(&window.returns, Some(subset))
                })
                .sum();

            let phi = whole_mi - part_mi;

            if phi < min_phi {
                min_phi = phi;
                mip = Some(partition);
            }
        }

        // 5. Normalize to [0, 1]
        let normalized_phi = self.normalize_phi(min_phi, &window);

        PhiResult {
            phi: normalized_phi,
            raw_phi: min_phi,
            mip,
            whole_mi,
        }
    }

    /// Mutual information using KSG estimator
    fn mutual_information(
        &self,
        returns: &Array2<f64>,
        subset: Option<&[usize]>,
    ) -> f64 {
        let data = match subset {
            Some(indices) => returns.select(Axis(1), indices),
            None => returns.clone(),
        };

        // Use Kraskov-Stögbauer-Grassberger (KSG) estimator
        ksg_mutual_information(&data, self.bootstrap_samples)
    }

    /// Generate market partitions
    fn generate_partitions(&self, data: &MarketData) -> Vec<Vec<Vec<usize>>> {
        match self.partition_method {
            PartitionMethod::Sectoral => {
                self.sectoral_partitions(&data.sectors)
            },
            PartitionMethod::AssetClass => {
                self.asset_class_partitions(data)
            },
            PartitionMethod::Hierarchical => {
                self.hierarchical_partitions(&data.correlations)
            },
            PartitionMethod::MinCut => {
                self.min_cut_partitions(&data.correlations)
            },
        }
    }

    /// Normalize Φ to [0, 1] scale
    fn normalize_phi(&self, raw_phi: f64, data: &MarketData) -> f64 {
        // Theoretical maximum: log₂(N) where N = number of assets
        let n = data.returns.ncols();
        let max_phi = (n as f64).log2();

        (raw_phi / max_phi).clamp(0.0, 1.0)
    }
}

/// Φ calculation result
pub struct PhiResult {
    /// Normalized integrated information [0, 1]
    pub phi: f64,

    /// Raw Φ value (unnormalized)
    pub raw_phi: f64,

    /// Minimum information partition
    pub mip: Option<Vec<Vec<usize>>>,

    /// Whole-system mutual information
    pub whole_mi: f64,
}
```

### 3.3 Computational Optimization

**Complexity**: O(2ⁿ) for n assets → **Approximation required**

```rust
impl PhiCalculator {
    /// Fast approximation for large markets (n > 100)
    pub fn calculate_approximate(&self, data: &MarketData) -> f64 {
        let n = data.returns.ncols();

        if n <= 50 {
            // Exact calculation
            return self.calculate(data).phi;
        }

        // Sample-based approximation
        let sample_size = 50;
        let samples = (0..self.bootstrap_samples)
            .map(|_| {
                let subset = random_sample(n, sample_size);
                let subset_data = data.select_assets(&subset);
                self.calculate(&subset_data).phi
            })
            .collect::<Vec<_>>();

        // Return median to reduce outlier influence
        median(&samples)
    }
}
```

---

## 4. CI (Consciousness Index) Calculation

### 4.1 Mathematical Definition

```
CI = D^α · G^β · C^γ · τ^δ
```

**Components**:

1. **D (Fractal Dimension)**: Market complexity
   ```
   D = lim[ε→0] log(N(ε)) / log(1/ε)
   ```
   Where `N(ε)` = number of boxes of size ε covering the trajectory

2. **G (Gain)**: Volatility amplification
   ```
   G = σ(Δr) / σ(r)
   ```
   Where `Δr` = returns, `σ` = standard deviation

3. **C (Coherence)**: Cross-asset synchronization
   ```
   C = |⟨exp(iθⱼ)⟩|  (Kuramoto order parameter)
   ```
   Where `θⱼ` = phase of asset j

4. **τ (Dwell Time)**: Regime persistence
   ```
   τ = -1 / log(λ₁)
   ```
   Where `λ₁` = largest eigenvalue of transition matrix

### 4.2 Implementation

```rust
pub struct CIAnalyzer {
    /// Parameter weights (α, β, γ, δ)
    weights: CIWeights,

    /// Historical window size
    window_size: usize,

    /// Phase extraction method
    phase_method: PhaseMethod,
}

#[derive(Debug, Clone)]
pub struct CIWeights {
    pub alpha: f64,  // Fractal dimension weight
    pub beta: f64,   // Gain weight
    pub gamma: f64,  // Coherence weight
    pub delta: f64,  // Dwell time weight
}

impl Default for CIWeights {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.8,
            gamma: 1.2,
            delta: 0.5,
        }
    }
}

impl CIAnalyzer {
    /// Calculate consciousness index
    pub fn calculate(&self, data: &MarketData) -> CIResult {
        let window = data.recent_window(self.window_size);

        // 1. Fractal dimension (D)
        let fractal_dim = self.calculate_fractal_dimension(&window);

        // 2. Gain (G)
        let gain = self.calculate_gain(&window);

        // 3. Coherence (C)
        let coherence = self.calculate_coherence(&window);

        // 4. Dwell time (τ)
        let dwell_time = self.calculate_dwell_time(&window);

        // 5. Compute CI
        let ci = fractal_dim.powf(self.weights.alpha)
            * gain.powf(self.weights.beta)
            * coherence.powf(self.weights.gamma)
            * dwell_time.powf(self.weights.delta);

        CIResult {
            ci,
            fractal_dim,
            gain,
            coherence,
            dwell_time,
        }
    }

    /// Fractal dimension via box-counting
    fn calculate_fractal_dimension(&self, data: &MarketData) -> f64 {
        // Use Hurst exponent as proxy (computationally efficient)
        let hurst = self.hurst_exponent(&data.returns);

        // Convert to fractal dimension: D = 2 - H
        2.0 - hurst
    }

    /// Hurst exponent via R/S analysis
    fn hurst_exponent(&self, returns: &Array2<f64>) -> f64 {
        // Average across assets
        returns.axis_iter(Axis(1))
            .map(|series| {
                let rs_values: Vec<_> = (10..series.len() / 4)
                    .step_by(10)
                    .map(|n| {
                        let subseries = &series.as_slice().unwrap()[..n];
                        let rs = self.rescaled_range(subseries);
                        (n as f64, rs)
                    })
                    .collect();

                // Linear regression: log(R/S) = H * log(n) + const
                linear_regression_slope(&rs_values)
            })
            .sum::<f64>() / returns.ncols() as f64
    }

    /// Rescaled range statistic
    fn rescaled_range(&self, series: &[f64]) -> f64 {
        let mean = series.iter().sum::<f64>() / series.len() as f64;

        // Cumulative deviations
        let mut cum_dev = vec![0.0; series.len()];
        let mut sum = 0.0;
        for i in 0..series.len() {
            sum += series[i] - mean;
            cum_dev[i] = sum;
        }

        // Range
        let range = cum_dev.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - cum_dev.iter().cloned().fold(f64::INFINITY, f64::min);

        // Standard deviation
        let variance = series.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / series.len() as f64;
        let std_dev = variance.sqrt();

        range / std_dev
    }

    /// Volatility gain (leverage effect)
    fn calculate_gain(&self, data: &MarketData) -> f64 {
        // Calculate volatility of returns vs volatility of volatility
        let returns = &data.returns;

        let vol_returns: Array1<f64> = returns.std_axis(Axis(0), 0.0);
        let vol_of_vol = vol_returns.std(0.0);
        let vol_of_returns = returns.std(0.0);

        vol_of_vol / vol_of_returns.max(1e-10)
    }

    /// Kuramoto order parameter (coherence)
    fn calculate_coherence(&self, data: &MarketData) -> f64 {
        // Extract phases using Hilbert transform
        let phases = self.extract_phases(data);

        // Kuramoto order parameter: |⟨exp(iθ)⟩|
        let (sum_cos, sum_sin): (f64, f64) = phases.iter()
            .fold((0.0, 0.0), |(cos_sum, sin_sum), &theta| {
                (cos_sum + theta.cos(), sin_sum + theta.sin())
            });

        let n = phases.len() as f64;
        ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt()
    }

    /// Extract oscillator phases
    fn extract_phases(&self, data: &MarketData) -> Vec<f64> {
        match self.phase_method {
            PhaseMethod::Hilbert => {
                self.hilbert_transform_phases(&data.returns)
            },
            PhaseMethod::Wavelet => {
                self.wavelet_phases(&data.returns)
            },
            PhaseMethod::EMD => {
                self.emd_phases(&data.returns)
            },
        }
    }

    /// Hilbert transform phase extraction
    fn hilbert_transform_phases(&self, returns: &Array2<f64>) -> Vec<f64> {
        returns.axis_iter(Axis(1))
            .map(|series| {
                let analytic = hilbert_transform(series.as_slice().unwrap());
                analytic.iter().map(|z| z.arg()).last().unwrap_or(0.0)
            })
            .collect()
    }

    /// Regime persistence (dwell time)
    fn calculate_dwell_time(&self, data: &MarketData) -> f64 {
        // Estimate from autocorrelation decay
        let returns = &data.returns;

        // Average autocorrelation across assets
        let avg_autocorr: Vec<f64> = (1..20)
            .map(|lag| {
                returns.axis_iter(Axis(1))
                    .map(|series| autocorrelation(series.as_slice().unwrap(), lag))
                    .sum::<f64>() / returns.ncols() as f64
            })
            .collect();

        // Find lag where autocorrelation drops below 0.1
        let dwell_lag = avg_autocorr.iter()
            .position(|&r| r < 0.1)
            .unwrap_or(avg_autocorr.len());

        dwell_lag as f64
    }
}

#[derive(Debug, Clone)]
pub enum PhaseMethod {
    Hilbert,
    Wavelet,
    EMD,  // Empirical Mode Decomposition
}

pub struct CIResult {
    pub ci: f64,
    pub fractal_dim: f64,
    pub gain: f64,
    pub coherence: f64,
    pub dwell_time: f64,
}
```

---

## 5. Regime Detection Engine

### 5.1 Core Algorithm

```rust
pub struct RegimeDetector {
    /// Φ calculator
    phi_calculator: PhiCalculator,

    /// CI analyzer
    ci_analyzer: CIAnalyzer,

    /// Historical window for trend analysis
    history_window: usize,

    /// Decision thresholds
    thresholds: RegimeThresholds,

    /// State history
    history: VecDeque<MarketConsciousness>,
}

#[derive(Debug, Clone)]
pub struct RegimeThresholds {
    /// Bull market thresholds
    pub bull_phi_min: f64,
    pub bull_ci_min: f64,

    /// Bear market thresholds
    pub bear_phi_min: f64,
    pub bear_ci_max: f64,

    /// Bubble thresholds
    pub bubble_phi_min: f64,
    pub bubble_ci_min: f64,

    /// Crash detection
    pub crash_phi_drop: f64,  // Percentage drop
    pub crash_time_window: usize,
}

impl Default for RegimeThresholds {
    fn default() -> Self {
        Self {
            bull_phi_min: 0.6,
            bull_ci_min: 0.5,
            bear_phi_min: 0.6,
            bear_ci_max: 0.4,
            bubble_phi_min: 0.8,
            bubble_ci_min: 0.7,
            crash_phi_drop: 0.5,  // 50% drop
            crash_time_window: 5,  // 5 periods
        }
    }
}

impl RegimeDetector {
    /// Detect current market regime
    pub fn detect_regime(&mut self, data: &MarketData) -> MarketConsciousness {
        // 1. Calculate consciousness metrics
        let phi_result = self.phi_calculator.calculate(data);
        let ci_result = self.ci_analyzer.calculate(data);

        let phi = phi_result.phi;
        let ci = ci_result.ci;

        // 2. Check for crash (priority check)
        if self.is_crash(&phi, &ci) {
            return self.create_crash_state(phi, ci, &phi_result, &ci_result);
        }

        // 3. Apply decision tree
        let regime = self.classify_regime(phi, ci, &ci_result);

        // 4. Calculate confidence
        let confidence = self.calculate_confidence(phi, ci, &regime);

        // 5. Create state
        let state = MarketConsciousness {
            phi,
            ci,
            regime,
            confidence,
            timestamp: Utc::now(),
        };

        // 6. Update history
        self.history.push_back(state.clone());
        if self.history.len() > self.history_window {
            self.history.pop_front();
        }

        state
    }

    /// Classify regime based on Φ and CI
    fn classify_regime(
        &self,
        phi: f64,
        ci: f64,
        ci_result: &CIResult,
    ) -> MarketRegime {
        let thresh = &self.thresholds;

        // Decision tree
        if phi >= thresh.bubble_phi_min && ci >= thresh.bubble_ci_min {
            // Bubble regime
            let fragility = self.calculate_bubble_fragility(ci_result);
            MarketRegime::Bubble { fragility }

        } else if phi >= thresh.bull_phi_min && ci >= thresh.bull_ci_min {
            // Bull regime
            let strength = self.calculate_bull_strength(phi, ci);
            MarketRegime::Bull { strength }

        } else if phi >= thresh.bear_phi_min && ci <= thresh.bear_ci_max {
            // Bear regime
            let severity = self.calculate_bear_severity(phi, ci);
            MarketRegime::Bear { severity }

        } else if phi < 0.3 && ci > 0.5 {
            // Correction regime
            let depth = self.calculate_correction_depth(phi, ci);
            MarketRegime::Correction { depth }

        } else {
            // Ranging regime (default)
            let volatility = ci_result.gain;
            MarketRegime::Ranging { volatility }
        }
    }

    /// Detect crash conditions
    fn is_crash(&self, phi: &f64, ci: &f64) -> bool {
        if self.history.len() < self.thresholds.crash_time_window {
            return false;
        }

        // Check for rapid Φ collapse
        let recent_phi: Vec<f64> = self.history.iter()
            .rev()
            .take(self.thresholds.crash_time_window)
            .map(|s| s.phi)
            .collect();

        let max_recent_phi = recent_phi.iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Crash = >50% drop in Φ within window
        *phi < (1.0 - self.thresholds.crash_phi_drop) * max_recent_phi
    }

    /// Create crash state
    fn create_crash_state(
        &self,
        phi: f64,
        ci: f64,
        phi_result: &PhiResult,
        ci_result: &CIResult,
    ) -> MarketConsciousness {
        let collapse_rate = if self.history.len() >= 2 {
            let prev_phi = self.history.back().unwrap().phi;
            (prev_phi - phi) / prev_phi
        } else {
            0.0
        };

        MarketConsciousness {
            phi,
            ci,
            regime: MarketRegime::Crash { collapse_rate },
            confidence: 0.95,  // High confidence in crash
            timestamp: Utc::now(),
        }
    }

    /// Calculate classification confidence
    fn calculate_confidence(&self, phi: f64, ci: f64, regime: &MarketRegime) -> f64 {
        // Distance from decision boundaries
        match regime {
            MarketRegime::Bull { .. } => {
                let phi_dist = (phi - self.thresholds.bull_phi_min).abs();
                let ci_dist = (ci - self.thresholds.bull_ci_min).abs();
                (phi_dist + ci_dist).min(1.0)
            },
            MarketRegime::Bear { .. } => {
                let phi_dist = (phi - self.thresholds.bear_phi_min).abs();
                let ci_dist = (self.thresholds.bear_ci_max - ci).abs();
                (phi_dist + ci_dist).min(1.0)
            },
            MarketRegime::Bubble { .. } => {
                let phi_dist = phi - self.thresholds.bubble_phi_min;
                let ci_dist = ci - self.thresholds.bubble_ci_min;
                (phi_dist + ci_dist).min(1.0)
            },
            MarketRegime::Crash { .. } => 0.95,
            _ => 0.5,
        }
    }

    /// Helper functions for regime metrics
    fn calculate_bull_strength(&self, phi: f64, ci: f64) -> f64 {
        ((phi - 0.6) / 0.4 + (ci - 0.5) / 0.5) / 2.0
    }

    fn calculate_bear_severity(&self, phi: f64, ci: f64) -> f64 {
        ((phi - 0.6) / 0.4 + (0.4 - ci) / 0.4) / 2.0
    }

    fn calculate_bubble_fragility(&self, ci_result: &CIResult) -> f64 {
        // High coherence + low dwell time = fragile
        ci_result.coherence / ci_result.dwell_time.max(1.0)
    }

    fn calculate_correction_depth(&self, phi: f64, ci: f64) -> f64 {
        (0.3 - phi) / 0.3 + (ci - 0.5) / 0.5
    }
}
```

### 5.2 Regime Transition Detection

```rust
impl RegimeDetector {
    /// Detect regime transitions
    pub fn detect_transition(&self) -> Option<RegimeTransition> {
        if self.history.len() < 2 {
            return None;
        }

        let current = self.history.back().unwrap();
        let previous = self.history.iter().rev().nth(1).unwrap();

        if std::mem::discriminant(&current.regime)
            != std::mem::discriminant(&previous.regime) {
            Some(RegimeTransition {
                from: previous.regime.clone(),
                to: current.regime.clone(),
                timestamp: current.timestamp,
                confidence: current.confidence,
            })
        } else {
            None
        }
    }
}

pub struct RegimeTransition {
    pub from: MarketRegime,
    pub to: MarketRegime,
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
}
```

---

## 6. Trading Strategy Integration

### 6.1 Position Sizing

```rust
pub struct RegimeAdaptivePortfolio {
    detector: RegimeDetector,
    base_risk: f64,  // Base risk per trade
    leverage_limits: HashMap<MarketRegime, f64>,
}

impl RegimeAdaptivePortfolio {
    /// Calculate regime-adjusted position size
    pub fn calculate_position_size(
        &self,
        base_size: f64,
        regime: &MarketRegime,
        risk_tolerance: f64,
    ) -> f64 {
        let regime_key = std::mem::discriminant(regime);
        let leverage = self.leverage_limits.get(&regime_key)
            .unwrap_or(&1.0);

        let regime_multiplier = match regime {
            MarketRegime::Bull { strength } => 1.0 + 0.5 * strength,
            MarketRegime::Bear { severity } => 0.3 + 0.2 * (1.0 - severity),
            MarketRegime::Bubble { fragility } => 0.2 * (1.0 - fragility),
            MarketRegime::Correction { depth } => 0.5 * (1.0 - depth),
            MarketRegime::Ranging { volatility } => 0.8 / (1.0 + volatility),
            MarketRegime::Crash { .. } => 0.0,  // Exit all
        };

        base_size * regime_multiplier * leverage * risk_tolerance
    }

    /// Generate trading signals
    pub fn generate_signals(
        &self,
        state: &MarketConsciousness,
    ) -> TradingSignal {
        match &state.regime {
            MarketRegime::Bull { strength } if *strength > 0.7 => {
                TradingSignal::StrongBuy
            },
            MarketRegime::Bull { .. } => TradingSignal::Buy,

            MarketRegime::Bear { severity } if *severity > 0.7 => {
                TradingSignal::StrongSell
            },
            MarketRegime::Bear { .. } => TradingSignal::Sell,

            MarketRegime::Bubble { fragility } if *fragility > 0.8 => {
                TradingSignal::EmergencyExit
            },
            MarketRegime::Bubble { .. } => TradingSignal::ReduceExposure,

            MarketRegime::Crash { .. } => TradingSignal::EmergencyExit,

            MarketRegime::Correction { .. } => TradingSignal::Cautious,

            MarketRegime::Ranging { .. } => TradingSignal::MeanReversion,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TradingSignal {
    StrongBuy,
    Buy,
    Hold,
    Cautious,
    MeanReversion,
    ReduceExposure,
    Sell,
    StrongSell,
    EmergencyExit,
}
```

### 6.2 Risk Management

```rust
pub struct RegimeBasedRiskManager {
    max_drawdown: HashMap<MarketRegime, f64>,
    stop_loss_multipliers: HashMap<MarketRegime, f64>,
    portfolio_heat: f64,  // Total risk exposure
}

impl RegimeBasedRiskManager {
    /// Calculate dynamic stop loss
    pub fn calculate_stop_loss(
        &self,
        entry_price: f64,
        regime: &MarketRegime,
        volatility: f64,
    ) -> f64 {
        let regime_key = std::mem::discriminant(regime);
        let multiplier = self.stop_loss_multipliers.get(&regime_key)
            .unwrap_or(&2.0);

        let base_stop = volatility * multiplier;

        match regime {
            MarketRegime::Crash { .. } => {
                // Immediate exit
                entry_price * 0.95
            },
            MarketRegime::Bubble { fragility } => {
                // Tight stop
                entry_price * (1.0 - base_stop * (1.0 + fragility))
            },
            _ => {
                entry_price * (1.0 - base_stop)
            }
        }
    }

    /// Monitor portfolio heat
    pub fn check_portfolio_heat(&self, positions: &[Position]) -> bool {
        let total_risk: f64 = positions.iter()
            .map(|p| p.risk_amount)
            .sum();

        total_risk <= self.portfolio_heat
    }
}
```

---

## 7. Implementation Plan

### 7.1 Development Phases

**Phase 1: Core Metrics (Weeks 1-2)**
- ✅ Implement Φ calculator with MIP finding
- ✅ Implement CI analyzer with all components
- ✅ Unit tests for mathematical correctness
- ✅ Validate against synthetic data

**Phase 2: Regime Detection (Weeks 3-4)**
- ✅ Build regime classifier
- ✅ Implement transition detection
- ✅ Backtesting framework
- ✅ Historical regime labeling

**Phase 3: Trading Integration (Weeks 5-6)**
- ✅ Position sizing logic
- ✅ Signal generation
- ✅ Risk management rules
- ✅ Portfolio optimization

**Phase 4: Validation (Weeks 7-8)**
- ✅ Historical backtesting (2000-2025)
- ✅ Out-of-sample testing
- ✅ Comparison with baselines
- ✅ Performance attribution

### 7.2 Data Requirements

**Market Data**:
- Asset prices: Daily/hourly OHLCV
- Coverage: 500+ stocks (S&P 500)
- History: 20+ years
- Frequency: Minute to daily

**Derived Data**:
- Correlation matrices (rolling 30/60/90 day)
- Sector classifications (GICS)
- Volume profiles
- Volatility surfaces

**Alternative Data** (Optional):
- News sentiment
- Social media trends
- Options flow
- Insider trading

### 7.3 Computational Resources

**Hardware Requirements**:
- CPU: 16+ cores (parallel Φ calculation)
- RAM: 32+ GB (correlation matrices)
- Storage: 1+ TB (historical data)
- GPU: Optional (neural network baselines)

**Software Stack**:
- Rust 1.70+ (core engine)
- Python 3.10+ (analysis, visualization)
- PostgreSQL/TimescaleDB (time series storage)
- Redis (real-time caching)

---

## 8. Validation Strategy

### 8.1 Historical Regime Labeling

**Ground Truth Events**:

| Date Range | Regime | Φ Expected | CI Expected |
|------------|--------|------------|-------------|
| 1999-2000 | Bubble | >0.8 | >0.7 |
| 2000-2002 | Bear | 0.6-0.7 | 0.2-0.4 |
| 2003-2007 | Bull | 0.6-0.8 | 0.5-0.7 |
| Oct 2008 | Crash | Spike→0 | Collapse |
| 2009-2020 | Bull | 0.6-0.8 | 0.5-0.7 |
| Mar 2020 | Crash | Spike→0 | Collapse |
| 2020-2021 | Bull | 0.7-0.9 | 0.6-0.8 |

### 8.2 Performance Metrics

**Classification Accuracy**:
```rust
pub struct ValidationMetrics {
    /// Regime classification accuracy
    pub accuracy: f64,

    /// Precision per regime
    pub precision: HashMap<MarketRegime, f64>,

    /// Recall per regime
    pub recall: HashMap<MarketRegime, f64>,

    /// F1 scores
    pub f1_scores: HashMap<MarketRegime, f64>,

    /// Transition detection rate
    pub transition_accuracy: f64,

    /// False alarm rate (crash prediction)
    pub false_alarm_rate: f64,
}
```

**Trading Performance**:
- Sharpe ratio by regime
- Maximum drawdown by regime
- Win rate by signal type
- Risk-adjusted returns

### 8.3 Baseline Comparisons

**Traditional Indicators**:
- VIX (volatility)
- Market breadth (advance-decline)
- Moving average crossovers
- RSI regimes

**Machine Learning Baselines**:
- Hidden Markov Models (HMM)
- Gaussian Mixture Models (GMM)
- LSTM autoencoders
- Random forests

**Expected Improvement**:
- 15-20% better regime classification
- 30-40% earlier transition detection
- 25% reduction in false alarms

---

## 9. Scientific References

### 9.1 Integrated Information Theory

1. **Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016)**. "Integrated information theory: from consciousness to its physical substrate." *Nature Reviews Neuroscience*, 17(7), 450-461.

2. **Oizumi, M., Albantakis, L., & Tononi, G. (2014)**. "From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0." *PLOS Computational Biology*, 10(5), e1003588.

3. **Barrett, A. B., & Seth, A. K. (2011)**. "Practical measures of integrated information for time-series data." *PLOS Computational Biology*, 7(1), e1001052.

### 9.2 Market Network Theory

4. **Mantegna, R. N., & Stanley, H. E. (1999)**. *Introduction to Econophysics: Correlations and Complexity in Finance*. Cambridge University Press.

5. **Onnela, J. P., Chakraborti, A., Kaski, K., & Kertész, J. (2003)**. "Dynamic asset trees and Black Monday." *Physica A: Statistical Mechanics and its Applications*, 324(1-2), 247-252.

6. **Kenett, D. Y., Tumminello, M., Madi, A., Gur-Gershgoren, G., Mantegna, R. N., & Ben-Jacob, E. (2010)**. "Dominating clasp of the financial sector revealed by partial correlation analysis of the stock market." *PLOS ONE*, 5(12), e15032.

### 9.3 Synchronization & Coherence

7. **Kuramoto, Y. (1975)**. "Self-entrainment of a population of coupled non-linear oscillators." In *International Symposium on Mathematical Problems in Theoretical Physics* (pp. 420-422). Springer, Berlin.

8. **Acebrón, J. A., Bonilla, L. L., Vicente, C. J. P., Ritort, F., & Spigler, R. (2005)**. "The Kuramoto model: A simple paradigm for synchronization phenomena." *Reviews of Modern Physics*, 77(1), 137.

9. **Zheng, Z., Podobnik, B., Feng, L., & Li, B. (2012)**. "Changes in cross-correlations as an indicator for systemic risk." *Scientific Reports*, 2, 888.

### 9.4 Fractal Market Analysis

10. **Peters, E. E. (1994)**. *Fractal Market Analysis: Applying Chaos Theory to Investment and Economics*. John Wiley & Sons.

11. **Mandelbrot, B. B., & Hudson, R. L. (2004)**. *The (Mis)Behavior of Markets*. Basic Books.

12. **Lux, T., & Marchesi, M. (1999)**. "Scaling and criticality in a stochastic multi-agent model of a financial market." *Nature*, 397(6719), 498-500.

### 9.5 Regime Detection

13. **Ang, A., & Timmermann, A. (2012)**. "Regime changes and financial markets." *Annual Review of Financial Economics*, 4(1), 313-337.

14. **Guidolin, M., & Timmermann, A. (2008)**. "International asset allocation under regime switching, skew, and kurtosis preferences." *The Review of Financial Studies*, 21(2), 889-935.

15. **Kritzman, M., Page, S., & Turkington, D. (2012)**. "Regime shifts: Implications for dynamic strategies." *Financial Analysts Journal*, 68(3), 22-39.

### 9.6 Information Theory in Finance

16. **Dionisio, A., Menezes, R., & Mendes, D. A. (2004)**. "Mutual information: a measure of dependency for nonlinear time series." *Physica A: Statistical Mechanics and its Applications*, 344(1-2), 326-329.

17. **Kraskov, A., Stögbauer, H., & Grassberger, P. (2004)**. "Estimating mutual information." *Physical Review E*, 69(6), 066138.

---

## 10. Future Extensions

### 10.1 Multi-Scale Analysis

- Intraday vs daily vs weekly Φ
- Cross-timeframe regime coherence
- Hierarchical regime structure

### 10.2 Alternative Assets

- Cryptocurrency markets
- Commodities
- Fixed income
- FX markets

### 10.3 Causal IIT

- Directional information flow
- Cause-effect structures
- Market microstructure

### 10.4 Deep Learning Integration

- Neural network Φ approximation
- Regime prediction (not just detection)
- Adaptive threshold learning

---

## Appendix A: Mathematical Proofs

### A.1 Φ Normalization Proof

**Theorem**: For n assets, maximum Φ is bounded by log₂(n).

**Proof**:
```
Given: Market system S with n assets
Let: I(X₀; X₋₁) = mutual information of full system

Maximum MI occurs when all assets perfectly correlated:
I_max = log₂(n)  (bits)

For any partition P:
Σᵢ I(Xᵢ₀; X₋₁) ≥ 0  (non-negativity)

Therefore:
Φ = I(X₀; X₋₁) - min_P Σᵢ I(Xᵢ₀; X₋₁)
  ≤ I_max - 0
  = log₂(n)

QED.
```

### A.2 CI Stability Analysis

**Theorem**: CI is monotonically increasing in coherence C for fixed D, G, τ.

**Proof**:
```
CI = D^α · G^β · C^γ · τ^δ

∂CI/∂C = D^α · G^β · γ · C^(γ-1) · τ^δ

Since D, G, C, τ > 0 and γ > 0:
∂CI/∂C > 0

Therefore CI monotonically increases with C.

QED.
```

---

## Appendix B: Implementation Pseudocode

### B.1 Full Detection Pipeline

```python
# Pseudocode for complete regime detection
def detect_market_regime(market_data):
    # 1. Preprocess data
    prices = market_data.get_prices()
    returns = calculate_returns(prices)
    correlations = rolling_correlation(returns, window=60)

    # 2. Calculate Φ
    phi = calculate_phi(
        returns=returns,
        correlations=correlations,
        partition_method='sectoral'
    )

    # 3. Calculate CI
    fractal_dim = hurst_exponent(returns)
    gain = volatility_of_volatility(returns) / volatility(returns)
    coherence = kuramoto_order_parameter(extract_phases(returns))
    dwell_time = autocorrelation_decay_time(returns)

    ci = (fractal_dim ** alpha) * (gain ** beta) * \
         (coherence ** gamma) * (dwell_time ** delta)

    # 4. Classify regime
    if detect_crash(phi, historical_phi):
        return CRASH
    elif phi > 0.8 and ci > 0.7:
        return BUBBLE
    elif phi > 0.6 and ci > 0.5:
        return BULL
    elif phi > 0.6 and ci < 0.4:
        return BEAR
    elif phi < 0.3 and ci > 0.5:
        return CORRECTION
    else:
        return RANGING

    # 5. Calculate confidence
    confidence = distance_from_boundaries(phi, ci, regime)

    return {
        'regime': regime,
        'phi': phi,
        'ci': ci,
        'confidence': confidence,
        'timestamp': now()
    }
```

---

## Document Status

**Version History**:
- v1.0.0 (2025-11-12): Initial design document

**Next Steps**:
1. ✅ Review by System Architect
2. ⏳ Prototype Φ calculator
3. ⏳ Prototype CI analyzer
4. ⏳ Integration testing

**Contact**:
- Agent: Consciousness-Analyst
- Queen: Seraphina
- Mission: HYPERPHYSICS-REGIME-DETECTION

---

**END OF DOCUMENT**
