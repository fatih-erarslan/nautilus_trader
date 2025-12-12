# QKS Plugin - Quantum Knowledge System Core

**Drop-in Rust crate exposing all 8 cognitive layers with FFI bindings for cross-language integration**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.91%2B-orange.svg)](https://www.rust-lang.org/)

## Overview

The QKS Plugin provides a scientifically-grounded cognitive architecture implementing 8 layers of computation, from thermodynamic optimization to full metacognitive agency. It exposes a thread-safe, FFI-compatible API for integration with C, C++, Python, and other languages.

## Architecture - 8 Cognitive Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 8: Integration                                            │
│   → Cognitive loop orchestration, homeostasis, system-wide     │
│      resource management                                        │
├─────────────────────────────────────────────────────────────────┤
│ Layer 7: Metacognition                                          │
│   → Self-modeling, strategy selection, meta-learning (MAML)    │
├─────────────────────────────────────────────────────────────────┤
│ Layer 6: Consciousness                                          │
│   → IIT Φ computation, global workspace theory                 │
├─────────────────────────────────────────────────────────────────┤
│ Layer 5: Collective Intelligence                                │
│   → Swarm coordination, distributed consensus (Raft, Byzantine)│
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Learning & Reasoning                                   │
│   → STDP plasticity, active inference, reasoning backends      │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: Decision Making                                        │
│   → Swarm intelligence (14 algorithms), VQE optimization       │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: Cognitive Architecture                                 │
│   → Holographic cortex, episodic memory (HNSW), attention      │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: Thermodynamic Optimization                             │
│   → Energy management, pBit dynamics, simulated annealing      │
└─────────────────────────────────────────────────────────────────┘
```

## Scientific Foundation

This plugin is built on peer-reviewed research:

1. **Integrated Information Theory (IIT)** - Tononi et al. (2016)
   *Nature Reviews Neuroscience* - DOI: 10.1038/nrn.2016.44

2. **Free Energy Principle (FEP)** - Friston (2010)
   *Nature Reviews Neuroscience* - DOI: 10.1038/nrn.2787

3. **Global Workspace Theory (GWT)** - Baars (1988)
   Consciousness mechanism via broadcast architecture

4. **Active Inference** - Friston et al. (2017)
   Perception-action loops via predictive coding

5. **Autopoiesis** - Maturana & Varela (1980)
   Self-organizing biological systems

## Quick Start (Rust)

```rust
use qks_plugin::{QksPlugin, QksConfigBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = QksConfigBuilder::new()
        .phi_threshold(1.5)
        .energy_setpoint(0.8)
        .meta_learning(true)
        .build();

    let mut plugin = QksPlugin::new(config);
    plugin.initialize()?;
    plugin.start()?;

    for i in 0..100 {
        let result = plugin.iterate()?;
        println!("Iteration {}: Φ={:.3}", i, result.phi);
    }

    plugin.shutdown()?;
    Ok(())
}
```

## Quick Start (C/C++)

```c
#include "qks_plugin.h"

int main() {
    QksHandle plugin = qks_create();
    qks_initialize(plugin);
    qks_start(plugin);

    double phi;
    qks_get_phi(plugin, &phi);
    printf("Φ = %.4f\n", phi);

    qks_destroy(plugin);
    return 0;
}
```

## Configuration

```rust
let config = QksConfigBuilder::new()
    .phi_threshold(1.5)
    .energy_setpoint(0.7)
    .meta_learning(true)
    .collective(true)
    .gpu(true)
    .build();
```

## License

Dual licensed under MIT OR Apache-2.0
