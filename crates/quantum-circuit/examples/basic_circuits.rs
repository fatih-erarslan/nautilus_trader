//! Basic quantum circuit examples
//! 
//! This example demonstrates how to create and execute basic quantum circuits
//! using the quantum-circuit crate.

use quantum_circuit::{
    Circuit, CircuitBuilder, gates::*, constants, utils,
    simulation::Simulator,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Quantum Circuit Basic Examples");
    println!("================================");
    
    // Example 1: Simple single-qubit gates
    single_qubit_examples()?;
    
    // Example 2: Two-qubit entanglement
    entanglement_examples()?;
    
    // Example 3: Circuit builder pattern
    circuit_builder_examples()?;
    
    // Example 4: Parameterized circuits
    parameterized_circuit_examples()?;
    
    // Example 5: Measurement and sampling
    measurement_examples()?;
    
    Ok(())
}

fn single_qubit_examples() -> quantum_circuit::Result<()> {
    println!("\n📊 Single-Qubit Gate Examples");
    println!("-----------------------------");
    
    // Create a single qubit in |0⟩ state
    let mut circuit = Circuit::new(1);
    
    // Apply Hadamard gate to create superposition
    circuit.add_gate(Box::new(Hadamard::new(0)))?;
    
    let state = circuit.execute()?;
    println!("After Hadamard gate:");
    print_quantum_state(&state, &["0", "1"]);
    
    // Apply Pauli-X gate
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Box::new(PauliX::new(0)))?;
    
    let state = circuit.execute()?;
    println!("\nAfter Pauli-X gate:");
    print_quantum_state(&state, &["0", "1"]);
    
    // Apply rotation gates
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Box::new(RY::new(0, PI / 3.0)))?; // 60 degrees
    
    let state = circuit.execute()?;
    println!("\nAfter RY(π/3) rotation:");
    print_quantum_state(&state, &["0", "1"]);
    
    Ok(())
}

fn entanglement_examples() -> quantum_circuit::Result<()> {
    println!("\n🔗 Two-Qubit Entanglement Examples");
    println!("----------------------------------");
    
    // Create Bell state: (|00⟩ + |11⟩) / √2
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Box::new(Hadamard::new(0)))?;
    circuit.add_gate(Box::new(CNOT::new(0, 1)))?;
    
    let state = circuit.execute()?;
    println!("Bell State |Φ+⟩ = (|00⟩ + |11⟩)/√2:");
    print_quantum_state(&state, &["00", "01", "10", "11"]);
    
    // Create different Bell state: (|01⟩ + |10⟩) / √2
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Box::new(Hadamard::new(0)))?;
    circuit.add_gate(Box::new(PauliX::new(1)))?;
    circuit.add_gate(Box::new(CNOT::new(0, 1)))?;
    
    let state = circuit.execute()?;
    println!("\nBell State |Φ-⟩ = (|01⟩ + |10⟩)/√2:");
    print_quantum_state(&state, &["00", "01", "10", "11"]);
    
    // GHZ state with 3 qubits: (|000⟩ + |111⟩) / √2
    let mut circuit = Circuit::new(3);
    circuit.add_gate(Box::new(Hadamard::new(0)))?;
    circuit.add_gate(Box::new(CNOT::new(0, 1)))?;
    circuit.add_gate(Box::new(CNOT::new(1, 2)))?;
    
    let state = circuit.execute()?;
    println!("\nGHZ State |GHZ⟩ = (|000⟩ + |111⟩)/√2:");
    print_quantum_state(&state, &["000", "001", "010", "011", "100", "101", "110", "111"]);
    
    Ok(())
}

fn circuit_builder_examples() -> quantum_circuit::Result<()> {
    println!("\n🏗️  Circuit Builder Pattern Examples");
    println!("-----------------------------------");
    
    // Build a quantum Fourier transform circuit (simplified)
    let circuit = CircuitBuilder::new(3)
        .h(0)
        .cnot(0, 1)
        .h(1)
        .cnot(0, 2)
        .cnot(1, 2)
        .h(2)
        .build();
    
    println!("Quantum Fourier Transform (3 qubits):");
    println!("  Gates: {}", circuit.gate_count());
    println!("  Depth: {}", circuit.depth());
    println!("  Parameters: {}", circuit.parameter_count());
    
    let state = circuit.execute()?;
    print_quantum_state(&state, &["000", "001", "010", "011", "100", "101", "110", "111"]);
    
    // Build a variational ansatz circuit
    let circuit = CircuitBuilder::new(2)
        .ry(0, PI / 4.0)
        .ry(1, PI / 6.0)
        .cnot(0, 1)
        .rx(0, PI / 8.0)
        .rx(1, PI / 3.0)
        .build();
    
    println!("\nVariational Ansatz Circuit:");
    println!("  Gates: {}", circuit.gate_count());
    println!("  Parameters: {}", circuit.parameter_count());
    
    let state = circuit.execute()?;
    print_quantum_state(&state, &["00", "01", "10", "11"]);
    
    Ok(())
}

fn parameterized_circuit_examples() -> quantum_circuit::Result<()> {
    println!("\n⚙️  Parameterized Circuit Examples");
    println!("---------------------------------");
    
    // Create a parameterized rotation circuit
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Box::new(RY::new(0, 0.0)))?; // Parameter 0
    circuit.add_gate(Box::new(RX::new(1, 0.0)))?; // Parameter 1
    circuit.add_gate(Box::new(CNOT::new(0, 1)))?;
    circuit.add_gate(Box::new(RZ::new(0, 0.0)))?; // Parameter 2
    
    // Execute with different parameter values
    let param_sets = vec![
        vec![0.0, 0.0, 0.0],
        vec![PI/2.0, PI/4.0, PI/6.0],
        vec![PI, PI/2.0, PI/3.0],
    ];
    
    for (i, params) in param_sets.iter().enumerate() {
        println!("\nParameter set {}: {:?}", i + 1, params);
        let state = circuit.execute_with_parameters(params)?;
        print_quantum_state(&state, &["00", "01", "10", "11"]);
    }
    
    // Compute expectation value with different parameters
    let observable = constants::pauli_z(); // Single-qubit Z measurement on first qubit
    
    println!("\nExpectation values ⟨Z₀⟩:");
    for (i, params) in param_sets.iter().enumerate() {
        let expectation = circuit.expectation_value_with_parameters(&observable, params)?;
        println!("  Parameter set {}: {:.4}", i + 1, expectation);
    }
    
    Ok(())
}

fn measurement_examples() -> quantum_circuit::Result<()> {
    println!("\n📏 Measurement and Sampling Examples");
    println!("-----------------------------------");
    
    // Create a Bell state
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Box::new(Hadamard::new(0)))?;
    circuit.add_gate(Box::new(CNOT::new(0, 1)))?;
    
    // Use simulator for measurements
    let mut sim = Simulator::new(2);
    sim.execute_circuit(&circuit)?;
    
    println!("Bell state prepared: (|00⟩ + |11⟩)/√2");
    
    // Sample measurements
    let samples = sim.sample_measurements(10)?;
    println!("\nSample measurements (10 shots):");
    for (i, sample) in samples.iter().enumerate() {
        println!("  Shot {}: |{}{} ⟩", i + 1, sample[0], sample[1]);
    }
    
    // Larger sampling to show probability distribution
    let large_samples = sim.sample_measurements(1000)?;
    let mut count_00 = 0;
    let mut count_01 = 0;
    let mut count_10 = 0;
    let mut count_11 = 0;
    
    for sample in large_samples {
        match (sample[0], sample[1]) {
            (0, 0) => count_00 += 1,
            (0, 1) => count_01 += 1,
            (1, 0) => count_10 += 1,
            (1, 1) => count_11 += 1,
            _ => unreachable!(),
        }
    }
    
    println!("\nProbability distribution (1000 shots):");
    println!("  |00⟩: {} ({:.1}%)", count_00, 100.0 * count_00 as f64 / 1000.0);
    println!("  |01⟩: {} ({:.1}%)", count_01, 100.0 * count_01 as f64 / 1000.0);
    println!("  |10⟩: {} ({:.1}%)", count_10, 100.0 * count_10 as f64 / 1000.0);
    println!("  |11⟩: {} ({:.1}%)", count_11, 100.0 * count_11 as f64 / 1000.0);
    
    // Expectation value measurements
    let pauli_z = constants::pauli_z();
    let expectation = sim.expectation_value(&pauli_z)?;
    println!("\n⟨Z₀⟩ expectation value: {:.6}", expectation);
    
    // Test with different states
    println!("\nTesting different quantum states:");
    
    // |+⟩ state
    let mut sim = Simulator::new(1);
    let mut plus_circuit = Circuit::new(1);
    plus_circuit.add_gate(Box::new(Hadamard::new(0)))?;
    sim.execute_circuit(&plus_circuit)?;
    
    let exp_x = sim.expectation_value(&constants::pauli_x())?;
    let exp_z = sim.expectation_value(&constants::pauli_z())?;
    println!("  |+⟩ state: ⟨X⟩ = {:.3}, ⟨Z⟩ = {:.3}", exp_x, exp_z);
    
    // |1⟩ state
    let mut sim = Simulator::new(1);
    let mut one_circuit = Circuit::new(1);
    one_circuit.add_gate(Box::new(PauliX::new(0)))?;
    sim.execute_circuit(&one_circuit)?;
    
    let exp_z = sim.expectation_value(&constants::pauli_z())?;
    println!("  |1⟩ state: ⟨Z⟩ = {:.3}", exp_z);
    
    Ok(())
}

fn print_quantum_state(state: &quantum_circuit::StateVector, basis_labels: &[&str]) {
    println!("  State vector:");
    for (i, amplitude) in state.iter().enumerate() {
        if i < basis_labels.len() && amplitude.norm() > 1e-10 {
            let magnitude = amplitude.norm();
            let phase = amplitude.arg();
            
            if phase.abs() < 1e-6 {
                println!("    |{}⟩: {:.4}", basis_labels[i], magnitude);
            } else {
                println!("    |{}⟩: {:.4} × e^(i{:.3})", basis_labels[i], magnitude, phase);
            }
        }
    }
    
    // Verify normalization
    let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    println!("  Normalization: {:.6}", norm_sqr);
}