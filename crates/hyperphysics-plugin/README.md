# HyperPhysics Plugin

Drop-in plugin for accessing HyperPhysics swarm intelligence from any Rust application.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
hyperphysics-plugin = { path = "../hyperphysics-plugin" }
# Or from crates.io when published:
# hyperphysics-plugin = "0.1"
```

### Simple Optimization

```rust
use hyperphysics_plugin::prelude::*;

fn main() {
    // One-liner optimization
    let result = HyperPhysics::quick_optimize(10, |x| {
        x.iter().map(|xi| xi * xi).sum()
    }).unwrap();
    
    println!("Best: {:?} = {}", result.solution, result.fitness);
}
```

### Builder Pattern

```rust
use hyperphysics_plugin::prelude::*;

fn main() {
    let result = HyperPhysics::optimize()
        .dimensions(10)
        .bounds(-100.0, 100.0)
        .strategy(Strategy::GreyWolf)
        .population(50)
        .iterations(1000)
        .minimize(|x| x.iter().map(|xi| xi * xi).sum())
        .unwrap();
    
    println!("Best fitness: {}", result.fitness);
    println!("Iterations: {}", result.metrics.iterations);
    println!("Time: {}ms", result.metrics.time_ms);
}
```

### Multi-Strategy Swarm

```rust
use hyperphysics_plugin::prelude::*;

fn main() {
    let result = SwarmBuilder::new()
        .agents(60)
        .dimensions(10)
        .bounds(-5.0, 5.0)
        .strategies(vec![
            Strategy::ParticleSwarm,
            Strategy::GreyWolf,
            Strategy::Whale,
        ])
        .topology(Topology::Hyperbolic)
        .iterations(500)
        .minimize(|x| x.iter().map(|xi| xi * xi).sum())
        .unwrap();
    
    println!("Best: {}", result.fitness);
    
    // See which strategy performed best
    for (strategy, perf) in &result.strategy_performance {
        println!("{:?}: {}", strategy, perf);
    }
}
```

### pBit Lattice

```rust
use hyperphysics_plugin::prelude::*;

fn main() {
    let mut lattice = LatticeBuilder::new()
        .dimensions_2d(32, 32)
        .temperature(2.0)
        .coupling(1.0)
        .build()
        .unwrap();
    
    // Anneal to ground state
    lattice.anneal(0.1, 500);
    
    let state = lattice.state();
    println!("Magnetization: {}", state.magnetization);
    println!("Energy: {}", state.energy);
}
```

### Benchmark Strategies

```rust
use hyperphysics_plugin::prelude::*;

fn main() {
    let results = HyperPhysics::benchmark(10, |x| {
        x.iter().map(|xi| xi * xi).sum()
    }).unwrap();
    
    // Results sorted by fitness (best first)
    for (strategy, result) in results {
        println!("{:?}: {} ({}ms)", strategy, result.fitness, result.metrics.time_ms);
    }
}
```

## Strategies

| Strategy | Inspiration | Best For |
|----------|-------------|----------|
| `ParticleSwarm` | Bird flocking | General purpose |
| `GreyWolf` | Wolf pack hunting | Fast convergence |
| `Whale` | Bubble-net feeding | Exploration |
| `Cuckoo` | Lévy flights | Escaping local optima |
| `DifferentialEvolution` | Evolution | Robust optimization |
| `Adaptive` | Auto-selection | Unknown problems |
| `Firefly` | Bioluminescence | Multimodal |
| `Bat` | Echolocation | Balance |
| `Genetic` | Natural selection | Discrete spaces |
| `BeeColony` | Waggle dance | Expensive functions |

## Topologies

| Topology | Description |
|----------|-------------|
| `Mesh` | Fully connected (default) |
| `Star` | Hub-and-spoke |
| `Ring` | Circular |
| `Hyperbolic` | Poincaré disk |
| `SmallWorld` | Watts-Strogatz |
| `ScaleFree` | Barabási-Albert |
| `Hierarchical` | Tree |

## Features

- `default` - Full functionality
- `parallel` - Rayon parallelization
- `serde` - Serialization support
- `async` - Async/await support

## Examples

```bash
cargo run --example optimize
cargo run --example swarm
```

## License

MIT OR Apache-2.0
