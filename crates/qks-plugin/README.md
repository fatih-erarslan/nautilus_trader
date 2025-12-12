# QKS Plugin - Quantum Knowledge System Unified API

**Drop-in super pill of wisdom** for GPU-accelerated quantum computing.

## Features

- 🚀 **Metal GPU**: High-performance quantum simulation (up to 30 qubits on AMD RX 6800 XT)
- 🐍 **PennyLane Integration**: Seamless Python quantum machine learning
- 🌀 **Swarm Optimization**: 14+ biomimetic algorithms (Grey Wolf, PSO, Whale, etc.)
- 📐 **Hyperbolic Geometry**: H^11 Lorentz model for hierarchical state embeddings
- 🎲 **pBit Dynamics**: Probabilistic computing with Boltzmann statistics
- 🧠 **STDP Learning**: Fibonacci multi-scale plasticity for adaptive circuits
- 🔐 **Post-Quantum Security**: Dilithium MCP integration

## Quick Start

### Installation

```bash
cd /Volumes/Tengritek/Ashina/quantum_knowledge_system/rust-core/crates/qks-plugin
cargo build --release --features full
```

### Rust Example

```rust
use qks_plugin::prelude::*;

fn main() -> Result<()> {
    // Create Metal-accelerated quantum device
    let device = QksDevice::metal(20)?; // 20 qubits on GPU

    // Initialize quantum state
    let mut state = device.create_state()?;
    state.hadamard(0)?;
    state.cnot(0, 1)?;

    // Measure
    let probs = state.measure()?;
    println!("Probabilities: {:?}", probs);

    Ok(())
}
```

### VQE Optimization with Swarm Intelligence

```rust
use qks_plugin::prelude::*;

fn main() -> Result<()> {
    let optimizer = QksOptimizer::grey_wolf()
        .dimensions(10)
        .population(30)
        .iterations(100);

    let result = optimizer.minimize(|params| {
        // Compute energy expectation value
        compute_hamiltonian_energy(params)
    })?;

    println!("Optimal energy: {}", result.fitness);
    println!("Parameters: {:?}", result.position);

    Ok(())
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    QKS PLUGIN v0.1.0                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Metal   │  │PennyLane │  │  Swarm   │  │Hyperbolic│        │
│  │   GPU    │  │  Python  │  │Optimizer │  │ Geometry │        │
│  │(30 qubits)│  │  Bridge  │  │(14 algo) │  │  (H^11)  │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │              │
│       └─────────────┼─────────────┼─────────────┘              │
│                     │             │                            │
│  ┌─────────────┐  ┌─▼─────────┐  ┌▼─────────┐  ┌──────────┐   │
│  │    VQE      │  │   pBit    │  │  STDP    │  │Dilithium │   │
│  │ Variational │  │  System   │  │ Learning │  │   MCP    │   │
│  └─────────────┘  └───────────┘  └──────────┘  └──────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Feature Flags

- `full` (default): All features enabled
- `parallel`: Rayon parallel processing
- `serde`: Serialization support
- `gpu`: Metal GPU acceleration (macOS only)
- `pennylane`: PennyLane Python integration
- `hyperphysics`: HyperPhysics capabilities
- `python`: PyO3 Python bindings

## Examples

### Metal GPU Benchmark

```bash
cargo run --release --example metal_benchmark --features gpu
```

### VQE Optimization

```bash
cargo run --release --example vqe_optimization --features "gpu,hyperphysics"
```

### PennyLane Circuit

```bash
cargo run --release --example pennylane_circuit --features "pennylane,python"
```

## Integration with Python

The plugin is designed to work seamlessly with the Python QKS API:

```python
import pennylane as qml
from qks.devices import HyperPhysicsQuantumDevice

dev = HyperPhysicsQuantumDevice(wires=20)

@qml.qnode(dev)
def circuit(params):
    for i in range(20):
        qml.RY(params[i], wires=i)
    for i in range(19):
        qml.CNOT(wires=[i, i+1])
    return qml.expval(qml.PauliZ(0))

# Optimize with swarm intelligence
result = dev.vqe_optimize(
    circuit,
    initial_params=np.random.uniform(0, 2*np.pi, 20),
    strategy="grey_wolf"
)
```

## Dependencies

### Required
- Rust 1.70+
- qks-simulator (local crate)

### Optional
- qks-metal (for Metal GPU support, macOS only)
- hyperphysics-plugin (for HyperPhysics capabilities)
- PyO3 (for Python bindings)

## Performance

- **Metal GPU**: ~772 gates/s for 20-qubit circuits
- **CPU Simulator**: ~2000 gates/s for 10-qubit circuits
- **VQE Convergence**: ~100 iterations for 10-parameter optimization

## License

MIT OR Apache-2.0

## Related Projects

- [Quantum Knowledge System](../../README.md)
- [HyperPhysics Plugin](../../../HyperPhysics/crates/hyperphysics-plugin/)
- [QKS MCP Server](../../tools/qks-mcp/)
