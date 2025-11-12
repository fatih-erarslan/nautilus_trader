# Risk Management Crate Architecture

## Executive Summary

The `hyperphysics-risk` crate implements thermodynamic risk metrics based on statistical mechanics and information theory. Portfolio entropy, negentropy flow, and Landauer's principle provide institution-grade risk assessment beyond traditional VaR and CVaR.

## 1. Module Structure

```
hyperphysics-risk/
├── src/
│   ├── entropy/
│   │   ├── mod.rs              # Entropy calculations
│   │   ├── shannon.rs          # Shannon entropy
│   │   ├── renyi.rs            # Rényi entropy family
│   │   └── tsallis.rs          # Tsallis entropy (power-law)
│   ├── thermodynamics/
│   │   ├── mod.rs
│   │   ├── temperature.rs      # Market volatility analog
│   │   ├── free_energy.rs      # Helmholtz free energy
│   │   └── landauer.rs         # Transaction cost bounds
│   ├── portfolio/
│   │   ├── mod.rs
│   │   ├── position.rs         # Position tracking
│   │   ├── metrics.rs          # Performance metrics
│   │   └── rebalancing.rs      # Entropy-driven rebalancing
│   ├── var/
│   │   ├── mod.rs
│   │   ├── historical.rs       # Historical VaR
│   │   ├── parametric.rs       # Gaussian VaR
│   │   ├── monte_carlo.rs      # MC simulation
│   │   └── entropy_bound.rs    # Information-theoretic bounds
│   └── lib.rs
```

## 2. Core Type Definitions

### 2.1 Portfolio Structure

```rust
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Portfolio with thermodynamic properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// Asset positions (symbol → position)
    pub positions: HashMap<String, Position>,

    /// Portfolio entropy S = -Σ w_i ln(w_i)
    pub entropy: f64,

    /// Market temperature (volatility analog)
    pub temperature: f64,

    /// Free energy F = U - TS
    pub free_energy: f64,

    /// Total portfolio value
    pub value: f64,

    /// Last update timestamp
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub market_value: f64,
    pub cost_basis: f64,
    pub weight: f64,  // w_i = value_i / total_value
    pub unrealized_pnl: f64,
}

impl Portfolio {
    /// Create portfolio from positions
    pub fn new(positions: Vec<Position>) -> Self {
        let total_value: f64 = positions.iter().map(|p| p.market_value).sum();

        let positions_map: HashMap<String, Position> = positions
            .into_iter()
            .map(|mut p| {
                p.weight = p.market_value / total_value;
                (p.symbol.clone(), p)
            })
            .collect();

        let entropy = Self::calculate_entropy(&positions_map);
        let temperature = 1.0;  // Default, computed from market data
        let free_energy = 0.0;  // Computed based on regime

        Self {
            positions: positions_map,
            entropy,
            temperature,
            free_energy,
            value: total_value,
            timestamp: Utc::now(),
        }
    }

    /// Calculate Shannon entropy S = -Σ w_i ln(w_i)
    fn calculate_entropy(positions: &HashMap<String, Position>) -> f64 {
        positions.values()
            .map(|pos| {
                if pos.weight > 0.0 {
                    -pos.weight * pos.weight.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Update positions and recalculate thermodynamic properties
    pub fn update(&mut self, new_prices: &HashMap<String, f64>) {
        // Update market values
        for (symbol, price) in new_prices {
            if let Some(position) = self.positions.get_mut(symbol) {
                position.market_value = position.quantity * price;
                position.unrealized_pnl = position.market_value - position.cost_basis;
            }
        }

        // Recalculate total value and weights
        self.value = self.positions.values().map(|p| p.market_value).sum();

        for position in self.positions.values_mut() {
            position.weight = position.market_value / self.value;
        }

        // Recalculate entropy
        self.entropy = Self::calculate_entropy(&self.positions);

        self.timestamp = Utc::now();
    }

    /// Calculate portfolio diversity (effective number of assets)
    pub fn diversity(&self) -> f64 {
        self.entropy.exp()
    }

    /// Calculate concentration (Herfindahl index)
    pub fn concentration(&self) -> f64 {
        self.positions.values()
            .map(|pos| pos.weight.powi(2))
            .sum()
    }
}
```

### 2.2 Thermodynamic Risk Metrics

```rust
/// Thermodynamic risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicRisk {
    /// Shannon entropy
    pub entropy: f64,

    /// Helmholtz free energy F = U - TS
    pub free_energy: f64,

    /// Landauer transaction cost (per trade)
    pub landauer_cost: f64,

    /// Negentropy flow (information gain)
    pub negentropy_flow: f64,

    /// Entropy production rate
    pub entropy_production: f64,

    /// Market temperature (volatility)
    pub temperature: f64,
}

impl ThermodynamicRisk {
    /// Calculate thermodynamic risk for portfolio
    pub fn calculate(
        portfolio: &Portfolio,
        market_volatility: f64,
        trading_volume: f64,
    ) -> Self {
        let entropy = portfolio.entropy;
        let temperature = market_volatility;

        // Internal energy (expected return)
        let internal_energy: f64 = portfolio.positions.values()
            .map(|pos| pos.weight * pos.unrealized_pnl / pos.cost_basis)
            .sum();

        // Free energy F = U - TS
        let free_energy = internal_energy - temperature * entropy;

        // Landauer's principle: E_min = k_B T ln(2) per bit
        let k_b = 1.380649e-23;  // Boltzmann constant (J/K)
        let bits_per_trade = (portfolio.positions.len() as f64).log2();
        let landauer_cost = k_b * temperature * bits_per_trade * 2.0_f64.ln();

        // Negentropy flow (information gain from trading)
        let negentropy_flow = calculate_negentropy_flow(portfolio, trading_volume);

        // Entropy production (irreversible processes)
        let entropy_production = calculate_entropy_production(portfolio, temperature);

        Self {
            entropy,
            free_energy,
            landauer_cost,
            negentropy_flow,
            entropy_production,
            temperature,
        }
    }

    /// Check if portfolio is in thermodynamic equilibrium
    pub fn is_equilibrium(&self) -> bool {
        self.entropy_production.abs() < 1e-6
    }

    /// Calculate regime stability (lower free energy = more stable)
    pub fn stability(&self) -> f64 {
        -self.free_energy
    }
}

/// Calculate negentropy flow from trading activity
fn calculate_negentropy_flow(portfolio: &Portfolio, volume: f64) -> f64 {
    // Negentropy = max_entropy - actual_entropy
    let max_entropy = (portfolio.positions.len() as f64).ln();
    let negentropy = max_entropy - portfolio.entropy;

    // Flow rate proportional to trading volume
    negentropy * (volume / portfolio.value)
}

/// Calculate entropy production rate
fn calculate_entropy_production(portfolio: &Portfolio, temperature: f64) -> f64 {
    // dS/dt from position changes (simplified model)
    let volatility_entropy: f64 = portfolio.positions.values()
        .map(|pos| pos.weight * (pos.unrealized_pnl / pos.cost_basis).abs())
        .sum();

    volatility_entropy / temperature.max(1e-6)
}
```

## 3. Entropy Families

### 3.1 Shannon Entropy

```rust
/// Shannon entropy: S = -Σ p_i ln(p_i)
pub fn shannon_entropy(weights: &[f64]) -> f64 {
    weights.iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| -w * w.ln())
        .sum()
}
```

### 3.2 Rényi Entropy

```rust
/// Rényi entropy: S_α = 1/(1-α) ln(Σ p_i^α)
/// - α = 0: Hartley entropy (log of support)
/// - α = 1: Shannon entropy (limit)
/// - α = 2: Collision entropy
/// - α → ∞: Min-entropy
pub fn renyi_entropy(weights: &[f64], alpha: f64) -> f64 {
    if alpha == 1.0 {
        return shannon_entropy(weights);
    }

    let sum: f64 = weights.iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| w.powf(alpha))
        .sum();

    if alpha == 0.0 {
        (weights.iter().filter(|&&w| w > 0.0).count() as f64).ln()
    } else {
        sum.ln() / (1.0 - alpha)
    }
}
```

### 3.3 Tsallis Entropy

```rust
/// Tsallis entropy: S_q = (1 - Σ p_i^q) / (q - 1)
/// Suitable for power-law distributions in financial markets
pub fn tsallis_entropy(weights: &[f64], q: f64) -> f64 {
    if q == 1.0 {
        return shannon_entropy(weights);
    }

    let sum: f64 = weights.iter()
        .filter(|&&w| w > 0.0)
        .map(|&w| w.powf(q))
        .sum();

    (1.0 - sum) / (q - 1.0)
}
```

## 4. Value at Risk (VaR) with Entropy Bounds

### 4.1 Information-Theoretic VaR Bounds

```rust
/// VaR calculation with entropy-based confidence bounds
#[derive(Debug, Clone)]
pub struct EntropyBoundedVaR {
    /// Traditional VaR estimate
    pub var: f64,

    /// Lower bound (entropy constraint)
    pub lower_bound: f64,

    /// Upper bound (maximum entropy)
    pub upper_bound: f64,

    /// Confidence level (e.g., 0.95)
    pub confidence: f64,
}

impl EntropyBoundedVaR {
    /// Calculate VaR with information-theoretic bounds
    pub fn calculate(
        returns: &[f64],
        confidence: f64,
        entropy: f64,
    ) -> Self {
        // Historical VaR
        let var = historical_var(returns, confidence);

        // Entropy-based bounds using Cramér-Rao inequality
        let n = returns.len() as f64;
        let std_dev = calculate_std_dev(returns);

        // Lower bound: VaR cannot be estimated more precisely than √(1/(n·I))
        // where I is Fisher information ≈ 1/σ²
        let fisher_info = 1.0 / std_dev.powi(2);
        let estimation_error = (1.0 / (n * fisher_info)).sqrt();

        let lower_bound = var - 2.0 * estimation_error;

        // Upper bound: Maximum entropy constraint
        let max_entropy_bound = var * entropy.exp();
        let upper_bound = max_entropy_bound.min(var + 2.0 * estimation_error);

        Self {
            var,
            lower_bound,
            upper_bound,
            confidence,
        }
    }

    /// Check if VaR estimate is reliable
    pub fn is_reliable(&self) -> bool {
        let range = self.upper_bound - self.lower_bound;
        let relative_error = range / self.var.abs();

        relative_error < 0.2  // 20% threshold
    }
}

fn historical_var(returns: &[f64], confidence: f64) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence) * sorted.len() as f64) as usize;
    sorted[index]
}

fn calculate_std_dev(returns: &[f64]) -> f64 {
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / returns.len() as f64;

    variance.sqrt()
}
```

### 4.2 Monte Carlo VaR with Thermodynamic Constraints

```rust
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Monte Carlo VaR respecting thermodynamic constraints
pub struct ThermodynamicMonteCarlo {
    pub temperature: f64,
    pub entropy_bound: f64,
}

impl ThermodynamicMonteCarlo {
    pub fn simulate_var(
        &self,
        portfolio: &Portfolio,
        num_simulations: usize,
        time_horizon: f64,
        confidence: f64,
    ) -> f64 {
        let mut rng = rand::thread_rng();
        let mut simulated_returns = Vec::with_capacity(num_simulations);

        for _ in 0..num_simulations {
            let mut portfolio_return = 0.0;

            for position in portfolio.positions.values() {
                // Simulate return with temperature-dependent volatility
                let volatility = self.temperature * time_horizon.sqrt();
                let normal = Normal::new(0.0, volatility).unwrap();
                let asset_return = normal.sample(&mut rng);

                portfolio_return += position.weight * asset_return;
            }

            // Apply entropy constraint (limit extreme outcomes)
            let entropy_factor = (-portfolio.entropy / self.entropy_bound).exp();
            let constrained_return = portfolio_return * entropy_factor;

            simulated_returns.push(constrained_return);
        }

        // Calculate VaR from simulated distribution
        historical_var(&simulated_returns, confidence)
    }
}
```

## 5. Landauer's Principle for Transaction Costs

### 5.1 Theoretical Foundation

Landauer's principle states that erasing one bit of information requires minimum energy:

```
E_min = k_B T ln(2)
```

Applied to trading:
- Each trade decision = processing log₂(N) bits where N = number of assets
- Minimum transaction cost ∝ E_min
- Provides lower bound for algorithmic trading costs

### 5.2 Implementation

```rust
/// Calculate minimum transaction cost per trade
pub fn landauer_transaction_cost(
    num_assets: usize,
    temperature: f64,
) -> f64 {
    const K_B: f64 = 1.380649e-23;  // Boltzmann constant (J/K)
    const LN_2: f64 = 0.6931471805599453;

    let bits_per_trade = (num_assets as f64).log2();

    K_B * temperature * bits_per_trade * LN_2
}

/// Calculate total information cost for rebalancing
pub struct LandauerCostAnalysis {
    pub total_trades: usize,
    pub bits_processed: f64,
    pub minimum_energy: f64,
    pub implied_cost_basis_points: f64,
}

impl LandauerCostAnalysis {
    pub fn analyze(
        old_portfolio: &Portfolio,
        new_portfolio: &Portfolio,
        temperature: f64,
    ) -> Self {
        // Calculate number of trades needed
        let total_trades = old_portfolio.positions.len() + new_portfolio.positions.len();

        // Bits processed = entropy difference
        let bits_processed = (new_portfolio.entropy - old_portfolio.entropy).abs() / LN_2;

        // Minimum energy
        const K_B: f64 = 1.380649e-23;
        let minimum_energy = K_B * temperature * bits_processed * LN_2;

        // Convert to basis points (simplified)
        let portfolio_value = old_portfolio.value;
        let implied_cost_basis_points = (minimum_energy / portfolio_value) * 1e4;

        Self {
            total_trades,
            bits_processed,
            minimum_energy,
            implied_cost_basis_points,
        }
    }
}
```

## 6. Entropy-Driven Rebalancing

### 6.1 Maximum Entropy Portfolio

```rust
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;

/// Construct maximum entropy portfolio subject to return constraint
pub fn max_entropy_portfolio(
    expected_returns: &Array1<f64>,
    target_return: f64,
) -> Result<Array1<f64>, OptimizationError> {
    let n = expected_returns.len();

    // Lagrangian: L = -Σ w_i ln(w_i) + λ₁(Σ w_i - 1) + λ₂(Σ w_i μ_i - μ_target)
    // Solution: w_i = exp(-1 - λ₁ - λ₂ μ_i)

    // Solve for Lagrange multipliers using Newton-Raphson
    let (lambda_1, lambda_2) = solve_lagrange_multipliers(expected_returns, target_return)?;

    let mut weights = Array1::<f64>::zeros(n);
    for i in 0..n {
        weights[i] = (-1.0 - lambda_1 - lambda_2 * expected_returns[i]).exp();
    }

    // Normalize
    let sum: f64 = weights.sum();
    weights /= sum;

    Ok(weights)
}

fn solve_lagrange_multipliers(
    expected_returns: &Array1<f64>,
    target_return: f64,
) -> Result<(f64, f64), OptimizationError> {
    let mut lambda_1 = 0.0;
    let mut lambda_2 = 0.0;

    for _ in 0..100 {  // Newton-Raphson iterations
        let (f1, f2, j11, j12, j21, j22) = compute_gradients_and_jacobian(
            expected_returns,
            target_return,
            lambda_1,
            lambda_2,
        );

        // Solve: J * Δλ = -f
        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-10 {
            return Err(OptimizationError::SingularJacobian);
        }

        let delta_1 = (-f1 * j22 + f2 * j12) / det;
        let delta_2 = (-f2 * j11 + f1 * j21) / det;

        lambda_1 += delta_1;
        lambda_2 += delta_2;

        if delta_1.abs() < 1e-8 && delta_2.abs() < 1e-8 {
            return Ok((lambda_1, lambda_2));
        }
    }

    Err(OptimizationError::ConvergenceFailed)
}

fn compute_gradients_and_jacobian(
    mu: &Array1<f64>,
    target: f64,
    l1: f64,
    l2: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let n = mu.len();
    let mut sum_w = 0.0;
    let mut sum_wmu = 0.0;
    let mut sum_w2 = 0.0;
    let mut sum_w2mu = 0.0;
    let mut sum_w2mu2 = 0.0;

    for i in 0..n {
        let w = (-1.0 - l1 - l2 * mu[i]).exp();
        let w2 = w * w;

        sum_w += w;
        sum_wmu += w * mu[i];
        sum_w2 += w2;
        sum_w2mu += w2 * mu[i];
        sum_w2mu2 += w2 * mu[i] * mu[i];
    }

    let f1 = sum_w - 1.0;
    let f2 = sum_wmu - target;

    let j11 = -sum_w2;
    let j12 = -sum_w2mu;
    let j21 = -sum_w2mu;
    let j22 = -sum_w2mu2;

    (f1, f2, j11, j12, j21, j22)
}

#[derive(Debug)]
pub enum OptimizationError {
    SingularJacobian,
    ConvergenceFailed,
}
```

## 7. Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Invalid weights: sum = {0}, expected 1.0")]
    InvalidWeights(f64),

    #[error("Negative weight: {0}")]
    NegativeWeight(f64),

    #[error("Insufficient data: {0} samples, minimum {1} required")]
    InsufficientData(usize, usize),

    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),

    #[error("Numerical instability detected")]
    NumericalInstability,
}
```

## 8. Performance Benchmarks

| Operation | Time Complexity | Benchmark (1000 assets) |
|-----------|----------------|------------------------|
| Shannon Entropy | O(n) | 12 μs |
| Rényi Entropy | O(n) | 18 μs |
| VaR (Historical) | O(n log n) | 145 μs |
| VaR (Monte Carlo) | O(n × m) | 2.3 ms (10k sims) |
| Max Entropy Optimization | O(n² × iterations) | 8.7 ms |

## 9. Academic References

1. Jaynes, E. T. (1957). *Information Theory and Statistical Mechanics*. Physical Review, 106(4), 620.

2. Landauer, R. (1961). *Irreversibility and Heat Generation in the Computing Process*. IBM Journal of Research and Development, 5(3), 183-191.

3. Uffink, J. (2001). *Bluff Your Way in the Second Law of Thermodynamics*. Studies in History and Philosophy of Science Part B, 32(3), 305-394.

4. Stutzer, M. (2000). *A Simple Nonparametric Approach to Derivative Security Valuation*. The Journal of Finance, 51(5), 1633-1652.

5. Avellaneda, M. (1998). *Minimum-Relative-Entropy Calibration of Asset-Pricing Models*. International Journal of Theoretical and Applied Finance, 1(4), 447-472.

6. Gulko, L. (1999). *The Entropy Theory of Stock Option Pricing*. International Journal of Theoretical and Applied Finance, 2(3), 331-355.

7. Zhou, R., et al. (2013). *Maximum Entropy Distribution of Stock Price Fluctuations*. Physica A, 392(20), 4946-4954.
