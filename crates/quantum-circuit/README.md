# Quantum-Enhanced Circuit Architectures

A comprehensive Rust library for quantum-inspired machine learning with PennyLane-compatible API design. This crate provides quantum-enhanced classical ML using efficient simulation of small quantum circuits (up to 20 qubits).

## 🚀 Features

- **🧠 Quantum-Enhanced Neural Networks**: Hybrid classical-quantum neural architectures
- **⚡ Variational Quantum Circuits (VQC)**: Parameterized quantum circuits with automatic differentiation
- **🔧 QAOA-Inspired Optimization**: Quantum-enhanced optimization algorithms
- **🎯 Quantum Feature Embeddings**: Amplitude, angle, and parametric embeddings for classical data
- **🔗 Quantum Kernels**: Quantum-enhanced kernel methods for machine learning
- **📊 Quantum Attention**: Quantum-enhanced attention mechanisms
- **🍃 PennyLane Compatibility**: Familiar API for PennyLane users
- **⚡ High Performance**: Efficient classical simulation with Rust optimizations

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
quantum-circuit = "0.1.0"
```

## 🔬 Scientific Foundation

This library implements quantum-enhanced machine learning algorithms based on:

- **Variational Quantum Algorithms**: Parameter-shift rule for gradient computation
- **Quantum Feature Maps**: Quantum data encoding strategies
- **Quantum Kernel Methods**: Quantum-enhanced similarity measures
- **Hybrid Quantum-Classical Networks**: Integration of quantum and classical processing

All algorithms use efficient classical simulation (no actual quantum hardware required) but provide quantum-enhanced computational advantages for machine learning tasks.

## 🏗️ Architecture

```
quantum-circuit/
├── src/
│   ├── lib.rs                  # Main library interface
│   ├── gates.rs                # Quantum gates (unitary matrices)
│   ├── circuit.rs              # Circuit builder and execution
│   ├── simulation.rs           # Classical quantum simulation
│   ├── optimization.rs         # Quantum-enhanced optimizers
│   ├── embeddings.rs           # Quantum feature embeddings
│   ├── neural.rs               # Hybrid quantum-classical networks
│   └── pennylane_compat.rs     # PennyLane-compatible API
├── examples/                   # Usage examples
├── tests/                      # Integration tests
└── benches/                    # Performance benchmarks
```

## 💡 Quick Start

### Basic Quantum Circuit

```rust
use quantum_circuit::{Circuit, gates::*};

// Create a 2-qubit Bell state circuit
let mut circuit = Circuit::new(2);
circuit.add_gate(Box::new(Hadamard::new(0)))?;
circuit.add_gate(Box::new(CNOT::new(0, 1)))?;

// Execute the circuit
let state = circuit.execute()?;
println!("Bell state created: {:?}", state);
```

### Variational Quantum Circuit

```rust
use quantum_circuit::{VariationalCircuit, EntanglementPattern};

// Create a variational circuit for machine learning
let mut vqc = VariationalCircuit::new(3, 2, EntanglementPattern::Circular);
vqc.build_random()?;

let circuit = vqc.circuit();
let state = circuit.execute()?;
```

### Quantum Feature Embedding

```rust
use quantum_circuit::embeddings::{AmplitudeEmbedding, NormalizationMethod};

// Embed classical data into quantum state
let embedding = AmplitudeEmbedding::new(4, NormalizationMethod::L2);
let classical_data = vec![0.6, 0.8, 0.0, 0.0];
let quantum_state = embedding.embed(&classical_data)?;
```

### Hybrid Neural Network

```rust
use quantum_circuit::neural::SimpleHybridNet;

// Create hybrid quantum-classical network
let mut net = SimpleHybridNet::new(4, 8, 2, 3); // input, hidden, output, qubits

// Train on data
let history = net.train(&train_x, &train_y, 100)?;
```

### PennyLane-Compatible API

```rust
use quantum_circuit::pennylane_compat::{device, QNodeBuilder};

// Create quantum device
let mut dev = device("default.qubit", 2)?;

// Build quantum circuit
let mut builder = QNodeBuilder::new(2);
builder.hadamard(0)?
       .cnot(0, 1)?
       .expectation(pauli_z(), "Z".to_string())?;

let qnode = builder.build();
let result = qnode.execute(&mut dev, None)?;
```

## 🧪 Examples

### Variational Quantum Eigensolver (VQE)

Find the ground state of a quantum Hamiltonian:

```bash
cargo run --example vqe_example
```

### Quantum Machine Learning

Quantum-enhanced classification and feature extraction:

```bash
cargo run --example quantum_ml
```

### PennyLane Integration

PennyLane-compatible quantum machine learning:

```bash
cargo run --example pennylane_demo
```

## 🎯 Use Cases

### 1. **Quantum-Enhanced Classical ML**
- Feature embedding in quantum-inspired spaces
- Quantum kernel methods for SVM
- Variational quantum classifiers

### 2. **Hybrid Neural Networks**
- Classical-quantum layer integration
- Quantum attention mechanisms
- Parametric quantum feature maps

### 3. **Optimization Problems**
- QAOA-inspired combinatorial optimization
- VQE for quantum chemistry simulations
- Quantum-enhanced training algorithms

### 4. **Research and Education**
- Quantum algorithm prototyping
- Educational quantum computing examples
- Quantum machine learning research

## ⚡ Performance

- **Efficient Simulation**: Classical simulation up to 20 qubits
- **Parallel Execution**: Batch circuit simulation
- **Memory Optimized**: Sparse state representations where beneficial
- **SIMD Support**: Vectorized quantum operations (optional feature)

### Benchmarks

Run benchmarks to see performance characteristics:

```bash
cargo bench
```

## 🔬 Supported Quantum Gates

### Single-Qubit Gates
- **Pauli Gates**: X, Y, Z
- **Hadamard**: H
- **Rotations**: RX(θ), RY(θ), RZ(θ)
- **Identity**: I

### Two-Qubit Gates
- **CNOT**: Controlled-X
- **CZ**: Controlled-Z
- **CRX**: Controlled rotation X

### Parametric Gates
- All rotation gates support automatic differentiation
- Parameter-shift rule for gradient computation
- Variational circuit optimization

## 📊 Quantum Embeddings

### Amplitude Embedding
```rust
let embedding = AmplitudeEmbedding::new(4, NormalizationMethod::L2);
let quantum_state = embedding.embed(&classical_data)?;
```

### Angle Embedding
```rust
let embedding = AngleEmbedding::new(4).with_scale(PI);
let quantum_state = embedding.embed(&classical_data)?;
```

### Parametric Embedding
```rust
let embedding = ParametricEmbedding::new(4, 3, 2) // features, qubits, layers
    .with_entanglement(EntanglementPattern::Circular);
```

## 🤖 Optimization Algorithms

### QAOA (Quantum Approximate Optimization Algorithm)
```rust
let mut qaoa = QAOAOptimizer::new(config, 2);
let result = qaoa.optimize(cost_function, &initial_params)?;
```

### VQE (Variational Quantum Eigensolver)
```rust
let mut vqe = VQEOptimizer::new(config, hamiltonian);
let result = vqe.optimize_circuit(&circuit, cost_fn, &params)?;
```

### Adam with Quantum Gradients
```rust
let mut adam = AdamOptimizer::new(config);
let result = adam.optimize_with_gradients(objective, gradient_fn, &params)?;
```

## 🧠 Neural Network Integration

### Quantum Linear Layer
```rust
let layer = QuantumLinearLayer::new(input_dim, output_dim, n_qubits);
```

### Quantum Attention
```rust
let attention = QuantumAttention::new(input_dim, attention_dim, n_heads, n_qubits);
```

### Hybrid Networks
```rust
// Combines quantum embeddings with classical processing
let net = SimpleHybridNet::new(input_dim, hidden_dim, output_dim, n_qubits);
```

## 🔍 Testing

Run all tests:

```bash
# Unit tests
cargo test

# Integration tests  
cargo test --test integration_tests

# With output
cargo test -- --nocapture
```

## 📈 Benchmarking

Performance benchmarks:

```bash
# All benchmarks
cargo bench

# Specific benchmark group
cargo bench gate_operations
cargo bench circuit_execution
cargo bench quantum_embeddings
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution:
- **New quantum gates** and operations
- **Advanced optimization algorithms**
- **Quantum error correction** techniques
- **Performance optimizations**
- **Documentation** and examples

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PennyLane**: Inspiration for the API design
- **Qiskit**: Quantum computing concepts and best practices  
- **Cirq**: Circuit optimization techniques
- **JAX**: Automatic differentiation approaches

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{quantum_circuit_rs,
  title = {Quantum-Enhanced Circuit Architectures},
  author = {SERRA Project Contributors},
  year = {2024},
  url = {https://github.com/ruvnet/serra/tree/main/crates/quantum-circuit}
}
```

## 🔗 Related Projects

- **[SERRA Project](https://github.com/ruvnet/serra)**: The parent meta-framework
- **[PennyLane](https://pennylane.ai/)**: Quantum machine learning library
- **[Qiskit](https://qiskit.org/)**: Open-source quantum computing framework

---

**Built with ❤️ for quantum-enhanced machine learning**