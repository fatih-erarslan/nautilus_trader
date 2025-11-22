# Python Bridge Integration Guide

## Overview

This document outlines the integration requirements for replacing mock implementations in `src/python_bridge.rs` with the production-grade `hyperphysics-finance` crate.

## Current Mock Implementations (TO BE REPLACED)

### 1. Order Book Processing (Lines 130-138)

**Current Mock:**
```rust
let state = OrderBookState {
    best_bid: snapshot.bids.first().map(|l| l.price),
    best_ask: snapshot.asks.first().map(|l| l.price),
    mid_price: None,  // MOCK
    spread: None,     // MOCK
    // ...
};
```

**Required Replacement:**
```rust
use hyperphysics_finance::{FinanceSystem, types::L2Snapshot};

// Convert Python input to hyperphysics-finance L2Snapshot
let snapshot = convert_to_finance_snapshot(bids, asks, timestamp)?;

// Process with real FinanceSystem
self.finance_system.process_snapshot(snapshot)?;

// Get real order book state
let state = self.finance_system.orderbook_state()
    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "Order book not initialized"
    ))?;
```

### 2. Risk Metrics Calculation (Lines 161-174)

**Current Mock:**
```rust
let metrics = RiskMetrics {
    var_95: 0.0,      // MOCK
    var_99: 0.0,      // MOCK
    expected_shortfall: 0.0,  // MOCK
    volatility: simple_calc,  // MOCK IMPLEMENTATION
    // ...
};
```

**Required Replacement:**
```rust
use hyperphysics_finance::{RiskEngine, RiskConfig};
use ndarray::Array1;

// Convert numpy array to ndarray
let returns = Array1::from(returns_slice.to_vec());

// Calculate real risk metrics using production algorithms
let metrics = self.risk_engine.calculate_metrics()
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        format!("Risk calculation failed: {}", e)
    ))?;
```

### 3. Black-Scholes Greeks (Lines 199-206)

**Current Mock:**
```rust
let greeks = Greeks {
    delta: 0.5,    // MOCK CONSTANT
    gamma: 0.01,   // MOCK CONSTANT
    vega: 0.2,     // MOCK CONSTANT
    theta: -0.05,  // MOCK CONSTANT
    rho: 0.1,      // MOCK CONSTANT
};
```

**Required Replacement:**
```rust
use hyperphysics_finance::risk::{OptionParams, calculate_black_scholes};

let params = OptionParams {
    spot,
    strike,
    rate: risk_free_rate,
    volatility,
    time_to_maturity: time_to_expiry,
};

let (call_price, greeks) = calculate_black_scholes(&params)
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
        format!("Greeks calculation failed: {}", e)
    ))?;
```

## Data Type Conversions

### Python → Rust

```rust
// Convert Python list of (price, qty) tuples to L2Snapshot
fn python_to_l2snapshot(
    bids: Vec<(f64, f64)>,
    asks: Vec<(f64, f64)>,
    symbol: String,
    timestamp_us: u64,
) -> Result<L2Snapshot, FinanceError> {
    use hyperphysics_finance::types::{Price, Quantity};

    let bid_levels: Vec<(Price, Quantity)> = bids.iter()
        .map(|(p, q)| Ok((Price::new(*p)?, Quantity::new(*q)?)))
        .collect::<Result<Vec<_>, FinanceError>>()?;

    let ask_levels: Vec<(Price, Quantity)> = asks.iter()
        .map(|(p, q)| Ok((Price::new(*p)?, Quantity::new(*q)?)))
        .collect::<Result<Vec<_>, FinanceError>>()?;

    Ok(L2Snapshot {
        symbol,
        timestamp_us,
        bids: bid_levels,
        asks: ask_levels,
    })
}
```

### Rust → Python

```rust
// Convert RiskMetrics to Python dict
fn metrics_to_pydict<'py>(
    py: Python<'py>,
    metrics: &RiskMetrics,
) -> PyResult<&'py PyDict> {
    let dict = PyDict::new(py);
    dict.set_item("var_95", metrics.var_95)?;
    dict.set_item("var_99", metrics.var_99)?;
    dict.set_item("cvar_95", metrics.cvar_95)?;
    dict.set_item("volatility", metrics.volatility)?;
    dict.set_item("sharpe_ratio", metrics.sharpe_ratio)?;
    dict.set_item("max_drawdown", metrics.max_drawdown)?;
    dict.set_item("skewness", metrics.skewness)?;
    dict.set_item("kurtosis", metrics.kurtosis)?;
    Ok(dict)
}
```

## PyFinanceSystem State Management

Add `FinanceSystem` to the Python wrapper:

```rust
#[pyclass(name = "HyperPhysicsSystem")]
pub struct PyFinanceSystem {
    finance_system: FinanceSystem,  // Add this field
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyFinanceSystem {
    #[new]
    fn new() -> PyResult<Self> {
        let finance_system = FinanceSystem::default();
        let runtime = Arc::new(Runtime::new()?);

        Ok(Self {
            finance_system,
            runtime,
        })
    }
}
```

## Testing Requirements

### Unit Tests

Create `tests/python_bridge_tests.rs`:

```rust
#[test]
fn test_orderbook_integration() {
    // Test that Python bindings correctly use FinanceSystem
    // instead of mocks
}

#[test]
fn test_risk_calculation_integration() {
    // Test VaR, CVaR, volatility calculations match
    // between Python and Rust
}

#[test]
fn test_greeks_integration() {
    // Test Black-Scholes Greeks calculation
    // matches academic benchmarks
}
```

### Integration Tests

Create `tests/freqtrade_integration.py`:

```python
import numpy as np
from hyperphysics_finance import HyperPhysicsSystem

def test_full_workflow():
    system = HyperPhysicsSystem()

    # Test order book processing
    bids = [[100.0, 1.0], [99.5, 2.0]]
    asks = [[101.0, 1.5], [101.5, 2.5]]

    result = system.process_orderbook(bids, asks)
    assert result['mid_price'] > 0
    assert result['spread'] > 0

    # Test risk calculation
    returns = np.random.normal(0, 0.02, 100)
    risk = system.calculate_risk(returns)
    assert risk['var_95'] > 0
    assert risk['var_99'] > risk['var_95']

    # Test Greeks
    greeks = system.calculate_greeks(
        spot=100.0,
        strike=100.0,
        volatility=0.2,
        time_to_expiry=1.0,
        risk_free_rate=0.05
    )
    assert 0 < greeks['delta'] < 1
    assert greeks['gamma'] > 0
```

## Performance Validation

### Benchmark Against QuantLib

Create `benches/quantlib_comparison.py`:

```python
import time
import numpy as np
from hyperphysics_finance import HyperPhysicsSystem
import QuantLib as ql

def benchmark_greeks():
    """Compare HyperPhysics vs QuantLib Greeks calculation"""

    # QuantLib setup
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()

    # HyperPhysics setup
    system = HyperPhysicsSystem()

    # Run 10,000 calculations
    start = time.time()
    for _ in range(10000):
        greeks = system.calculate_greeks(
            spot=100.0,
            strike=100.0,
            volatility=0.2,
            time_to_expiry=1.0,
            risk_free_rate=0.05
        )
    hp_time = time.time() - start

    print(f"HyperPhysics: {hp_time:.3f}s for 10k calculations")
    # Target: <1ms per calculation = <10s total
    assert hp_time < 10.0
```

## Dependency Updates

Update `Cargo.toml` to include `hyperphysics-finance`:

```toml
[dependencies]
hyperphysics-finance = { path = "crates/hyperphysics-finance" }
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"
ndarray = "0.15"
```

## Migration Checklist

- [ ] Add `hyperphysics-finance` dependency to main crate
- [ ] Update `PyFinanceSystem` struct to include `FinanceSystem`
- [ ] Replace `process_orderbook` mock implementation
- [ ] Replace `calculate_risk` mock implementation
- [ ] Replace `calculate_greeks` mock implementation
- [ ] Add data type conversion helpers
- [ ] Create Python integration tests
- [ ] Run performance benchmarks vs QuantLib
- [ ] Update Python documentation
- [ ] Remove ALL mock data generators

## Validation Criteria

All implementations must meet:

1. **Correctness**: Match academic benchmarks from Hull (2018)
2. **Performance**: <1ms for Greeks, <10ms for VaR
3. **Coverage**: 100% test coverage
4. **No Mocks**: Zero tolerance for mock/synthetic data

## References

- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Hull, J. (2018). "Options, Futures, and Other Derivatives" (10th ed.)
- RiskMetrics (1996). "Technical Document", JP Morgan
- Jorion, P. (2006). "Value at Risk: The New Benchmark"
