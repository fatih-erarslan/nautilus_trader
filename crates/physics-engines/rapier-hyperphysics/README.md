# Rapier-HyperPhysics Integration

Physics-based market simulation for HFT using the Rapier3D physics engine.

## Overview

This crate bridges Rapier3D rigid body physics with HyperPhysics market modeling. It maps market entities (orders, participants, liquidity) to physics objects, runs simulation to model market dynamics, and extracts actionable trading signals from the physics results.

## Architecture

```
MarketTick → MarketMapper → Rapier Bodies → PhysicsSimulator → SignalExtractor → TradingSignal
```

### Components

1. **MarketMapper** (`market_mapper.rs`)
   - Maps order book to rigid bodies
   - Bid orders positioned left (negative X), ask orders right (positive X)
   - Y-axis represents price, mass represents volume
   - Market participants become larger bodies with forces

2. **PhysicsSimulator** (`physics_simulator.rs`)
   - Runs Rapier simulation with configurable timesteps
   - Tracks energy (volatility), momentum (direction), convergence
   - Supports external forces (market shocks)
   - Target: <500μs per cycle

3. **SignalExtractor** (`signal_extractor.rs`)
   - Analyzes bid/ask momentum differences
   - Classifies signals: StrongBuy, Buy, Hold, Sell, StrongSell
   - Estimates price movement and confidence
   - Identifies market regimes: Trending, Ranging, Breakout, etc.

## Usage

```rust
use rapier_hyperphysics::*;

// Create adapter
let mut adapter = RapierHyperPhysicsAdapter::new()
    .with_timestep(0.001);  // 1ms for sub-millisecond latency

// Map market to physics
let market_state = MarketState {
    bids: vec![(100.0, 10.0), (99.5, 15.0)],
    asks: vec![(100.5, 12.0), (101.0, 10.0)],
    mid_price: 100.25,
    volatility: 0.02,
    // ...
};

let mapper = MarketMapper::new();
let mapping = mapper.map_to_physics(
    &market_state,
    adapter.rigid_bodies_mut(),
    adapter.colliders_mut(),
)?;

// Run simulation
let simulator = PhysicsSimulator::new();
let sim_result = simulator.simulate(&mut adapter)?;

// Extract signal
let extractor = SignalExtractor::new();
let signal = extractor.extract_signal(&adapter, &mapping)?;

match signal.signal {
    TradingSignal::StrongBuy => println!("BUY (confidence: {})", signal.confidence),
    TradingSignal::StrongSell => println!("SELL (confidence: {})", signal.confidence),
    _ => println!("HOLD"),
}
```

## Performance Targets

- **Latency**: <500μs per full cycle (map → simulate → extract)
- **Throughput**: 2000+ cycles/second
- **Memory**: <10MB per simulation
- **Determinism**: Optional via feature flag

## Physics Mapping

### Order Book → Rigid Bodies
- **Bids**: X < 0 (left side), mass = volume
- **Asks**: X > 0 (right side), mass = volume  
- **Price**: Y-axis position
- **Spread**: Distance between bid/ask clusters

### Market Forces
- **Gravity**: Represents market trend/momentum
- **Damping**: Market friction/stabilization
- **Restitution**: Volatility (bounciness)
- **Collisions**: Order matching events

### Signal Derivation
- **Bid Momentum > Ask Momentum** → Bullish signal
- **Ask Momentum > Bid Momentum** → Bearish signal
- **High Total Energy** → High volatility
- **Low Energy + Strong Momentum** → Trending market
- **High Energy + Weak Momentum** → Ranging/choppy market

## Integration with HFT Ecosystem

This crate integrates with `hyperphysics-hft-ecosystem` via the `physics-rapier` feature:

```toml
hyperphysics-hft-ecosystem = { features = ["physics-rapier"] }
```

The `PhysicsEngineRouter` automatically routes market ticks to Rapier when configured:

```rust
let ecosystem = HFTEcosystem::builder()
    .with_physics_engine(PhysicsEngine::Rapier)
    .with_biomimetic_tier(BiomimeticTier::Tier1)
    .build()
    .await?;

let decision = ecosystem.execute_cycle(&market_tick).await?;
```

## Features

- `default`: Includes SIMD optimizations
- `simd`: Enable SIMD for faster physics
- `deterministic`: Deterministic simulation (for replay/auditing)

## Testing

```bash
cargo test -p rapier-hyperphysics
```

## Benchmarking

```bash
cargo bench -p rapier-hyperphysics
```

## Dependencies

- `rapier3d 0.22`: Physics engine with SIMD support
- `nalgebra 0.33`: Linear algebra
- `hyperphysics-core`: Core HyperPhysics types
- `hyperphysics-geometry`: Hyperbolic geometry
- `hyperphysics-market`: Market modeling

## License

MIT OR Apache-2.0
