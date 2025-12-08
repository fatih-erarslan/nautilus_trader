# HyperPhysics Swarm Intelligence

Meta-swarm system with pBit lattice, biomimetic strategies, and emergent intellect evolution.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    META-SWARM INTELLIGENCE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐                                                │
│  │   pBit LATTICE  │  Ising model + STDP + Boltzmann sampling       │
│  │  (Computational │  ┌────┬────┬────┐                              │
│  │     Fabric)     │  │ +1 │ -1 │ +1 │                              │
│  └────────┬────────┘  ├────┼────┼────┤                              │
│           │           │ -1 │ +1 │ -1 │                              │
│           ▼           └────┴────┴────┘                              │
│  ┌─────────────────┐                                                │
│  │   TOPOLOGIES    │  Star | Ring | Mesh | Hyperbolic | SmallWorld  │
│  │                 │  ScaleFree | Hierarchical | Dynamic            │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │   STRATEGIES    │  14+ biomimetic algorithms                     │
│  │                 │                                                │
│  │  SWARM          │  PSO (birds) | ACO (ants) | ABC (bees)         │
│  │  PACK           │  GWO (wolves) | WOA (whales)                   │
│  │  SCHOOL         │  Fish schooling | Salp chains                  │
│  │  FLOCK          │  Firefly | Bat | Cuckoo | MothFlame            │
│  │  COLONY         │  Bacterial | Social Spider                     │
│  │  EVOLUTION      │  GA | DE | Quantum PSO                         │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │   EVOLUTION     │  Genome | Fitness | Crossover | Mutation       │
│  │    ENGINE       │  Multi-objective | Pareto front                │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │   EMERGENT      │  Knowledge Graph | Insights | Recommendations  │
│  │   INTELLECT     │  Strategy affinities | Problem models          │
│  └─────────────────┘                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. pBit SpatioTemporal Lattice

Probabilistic computing fabric based on:
- **Ising model** for spin interactions
- **Boltzmann sampling** for probabilistic state updates
- **STDP** (Spike-Timing Dependent Plasticity) for learning
- **Annealing** for optimization

```rust
use hyperphysics_swarm_intelligence::lattice::{PBitLattice, LatticeConfig};

let config = LatticeConfig {
    dimensions: (16, 16, 4),
    temperature: 1.0,
    coupling_strength: 1.0,
    ..Default::default()
};

let mut lattice = PBitLattice::new(config)?;
lattice.anneal(0.01, 1000); // Quench to ground state
```

### 2. Swarm Topologies

8+ network topologies for agent organization:

| Topology | Description | Best For |
|----------|-------------|----------|
| **Star** | Hub-and-spoke | Fast broadcast |
| **Ring** | Circular | Ordered updates |
| **Mesh** | Fully connected | Small swarms |
| **Hierarchical** | Tree structure | Large scale |
| **Hyperbolic** | Poincaré disk | Hierarchical + local |
| **SmallWorld** | Watts-Strogatz | Short paths |
| **ScaleFree** | Barabási-Albert | Robust networks |
| **Dynamic** | Evolving | Adaptive |

### 3. Biomimetic Strategies

14+ animal-inspired optimization algorithms:

```rust
use hyperphysics_swarm_intelligence::strategy::{BiomimeticStrategy, StrategyConfig, StrategyType};

let config = StrategyConfig {
    strategy_type: StrategyType::GreyWolf,
    population_size: 50,
    max_iterations: 1000,
    bounds: vec![(-100.0, 100.0); 10],
    ..Default::default()
};

let mut strategy = BiomimeticStrategy::new(config)?;
let result = strategy.optimize(|x| x.iter().map(|xi| xi * xi).sum())?;
```

### 4. Evolution Engine

Evolve optimal strategy configurations:

```rust
use hyperphysics_swarm_intelligence::evolution::{EvolutionEngine, EvolutionConfig};

let config = EvolutionConfig {
    population_size: 50,
    max_generations: 100,
    elite_count: 5,
    ..Default::default()
};

let mut engine = EvolutionEngine::new(config);
let best_genome = engine.evolve(|genome| evaluate(genome))?;
```

### 5. Emergent Intellect

Record, learn, and recommend:

```rust
use hyperphysics_swarm_intelligence::intellect::{EmergentIntellect, IntellectRecord};

let mut intellect = EmergentIntellect::new();

// Record a successful run
intellect.record(record);

// Get recommendations for a new problem
let recommendations = intellect.recommend(&problem_signature);
```

## Features

- `default` - Parallel execution + lattice
- `parallel` - Rayon-based parallelization
- `lattice` - pBit computational fabric
- `quantum` - Quantum-enhanced algorithms
- `gpu` - GPU acceleration (wgpu)
- `full` - All features

## Tests

```bash
cargo test
# 15 tests pass
```

## Usage with HyperPhysics

This crate integrates with:
- `hyperphysics-optimization` - Algorithm implementations
- `bio-inspired-workspace` - Additional algorithms
- `autopoiesis` - Consciousness and natural drift

## License

MIT OR Apache-2.0
