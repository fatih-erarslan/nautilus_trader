//! Variational Quantum Eigensolver (VQE) example
//!
//! This example demonstrates how to use VQE to find the ground state
//! of a simple quantum system (hydrogen molecule simulation).

use quantum_circuit::{
    Circuit, CircuitBuilder, VariationalCircuit, EntanglementPattern,
    optimization::{VQEOptimizer, OptimizerConfig, VariationalOptimizer},
    simulation::Simulator,
    gates::*,
    constants,
};
use ndarray::Array2;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Variational Quantum Eigensolver (VQE) Example");
    println!("================================================");
    
    // Example 1: Simple 1-qubit system (spin-1/2 in magnetic field)
    single_qubit_vqe()?;
    
    // Example 2: 2-qubit Heisenberg model
    heisenberg_model_vqe()?;
    
    // Example 3: Hydrogen molecule (simplified)
    hydrogen_molecule_vqe()?;
    
    Ok(())
}

fn single_qubit_vqe() -> quantum_circuit::Result<()> {
    println!("\n🔬 Single-Qubit VQE: Spin in Magnetic Field");
    println!("-------------------------------------------");
    
    // Hamiltonian: H = -B_z * Z (magnetic field in z-direction)
    let magnetic_field = 1.0;
    let hamiltonian = -magnetic_field * constants::pauli_z();
    
    println!("Hamiltonian: H = -{} × σ_z", magnetic_field);
    println!("Theoretical ground state: |1⟩ with energy = -{}", magnetic_field);
    
    // Create VQE optimizer
    let config = OptimizerConfig {
        max_iterations: 100,
        tolerance: 1e-8,
        learning_rate: 0.1,
        verbose: true,
        ..Default::default()
    };
    
    let mut vqe = VQEOptimizer::new(config, hamiltonian.clone());
    
    // Ansatz: single-qubit rotation
    let ansatz = |theta: f64| -> Circuit {
        CircuitBuilder::new(1)
            .ry(0, theta)
            .build()
    };
    
    // Cost function
    let cost_function = |circuit: &Circuit, params: &[f64]| -> quantum_circuit::Result<f64> {
        let energy = circuit.expectation_value_with_parameters(&hamiltonian, params)?;
        Ok(energy)
    };
    
    // Initial parameters (start from |0⟩ state)
    let initial_params = vec![0.0];
    
    println!("\nStarting VQE optimization...");
    let result = vqe.optimize_circuit(&ansatz(0.0), cost_function, &initial_params)?;
    
    println!("\nVQE Results:");
    println!("  Optimal parameters: {:?}", result.optimal_params);
    println!("  Ground state energy: {:.8}", result.optimal_value);
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.function_evaluations);
    
    // Compare with analytical solution
    println!("  Theoretical energy: {:.8}", -magnetic_field);
    println!("  Error: {:.2e}", (result.optimal_value + magnetic_field).abs());
    
    // Show the final state
    let final_circuit = ansatz(result.optimal_params[0]);
    let final_state = final_circuit.execute()?;
    
    println!("\nFinal quantum state:");
    println!("  |0⟩ amplitude: {:.6}", final_state[0].norm());
    println!("  |1⟩ amplitude: {:.6}", final_state[1].norm());
    
    if result.optimal_params[0].abs() > PI - 0.1 {
        println!("  ✅ Converged to |1⟩ state (ground state)");
    } else {
        println!("  ⚠️  Converged to intermediate state");
    }
    
    Ok(())
}

fn heisenberg_model_vqe() -> quantum_circuit::Result<()> {
    println!("\n🧲 Two-Qubit Heisenberg Model VQE");
    println!("---------------------------------");
    
    // Heisenberg Hamiltonian: H = J(XX + YY + ZZ) where J < 0 (ferromagnetic)
    let coupling = -0.5;
    
    // Construct Hamiltonian matrices
    let xx = tensor_product(&constants::pauli_x(), &constants::pauli_x());
    let yy = tensor_product(&constants::pauli_y(), &constants::pauli_y());
    let zz = tensor_product(&constants::pauli_z(), &constants::pauli_z());
    let hamiltonian = coupling * (xx + yy + zz);
    
    println!("Hamiltonian: H = {} × (σ_x ⊗ σ_x + σ_y ⊗ σ_y + σ_z ⊗ σ_z)", coupling);
    println!("Theoretical ground state: |00⟩ or |11⟩ (ferromagnetic alignment)");
    
    // Create VQE optimizer
    let config = OptimizerConfig {
        max_iterations: 150,
        tolerance: 1e-8,
        learning_rate: 0.05,
        verbose: false, // Reduce output for cleaner display
        ..Default::default()
    };
    
    let mut vqe = VQEOptimizer::new(config, hamiltonian.clone());
    
    // Hardware-efficient ansatz
    let ansatz = |params: &[f64]| -> Circuit {
        let mut circuit = Circuit::new(2);
        
        // Layer 1: Single-qubit rotations
        circuit.add_gate(Box::new(RY::new(0, params[0]))).unwrap();
        circuit.add_gate(Box::new(RY::new(1, params[1]))).unwrap();
        
        // Entangling gate
        circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
        
        // Layer 2: More rotations
        circuit.add_gate(Box::new(RY::new(0, params[2]))).unwrap();
        circuit.add_gate(Box::new(RY::new(1, params[3]))).unwrap();
        
        circuit
    };
    
    // Cost function
    let cost_function = |circuit: &Circuit, params: &[f64]| -> quantum_circuit::Result<f64> {
        let energy = circuit.expectation_value_with_parameters(&hamiltonian, params)?;
        Ok(energy)
    };
    
    // Try different initial parameters
    let initial_parameter_sets = vec![
        vec![0.0, 0.0, 0.0, 0.0],           // |00⟩ state
        vec![PI, PI, 0.0, 0.0],             // |11⟩ state  
        vec![PI/2.0, PI/2.0, 0.0, 0.0],    // Random superposition
        vec![PI/4.0, 3.0*PI/4.0, PI/6.0, PI/3.0], // More random
    ];
    
    let mut best_result = None;
    let mut best_energy = f64::INFINITY;
    
    println!("\nTrying different initial states...");
    
    for (i, initial_params) in initial_parameter_sets.iter().enumerate() {
        println!("  Initial parameters {}: {:?}", i + 1, initial_params);
        
        let result = vqe.optimize_circuit(&ansatz(&[0.0; 4]), cost_function, initial_params)?;
        
        println!("    Final energy: {:.8}", result.optimal_value);
        println!("    Iterations: {}", result.iterations);
        
        if result.optimal_value < best_energy {
            best_energy = result.optimal_value;
            best_result = Some(result);
        }
    }
    
    let best_result = best_result.unwrap();
    
    println!("\nBest VQE Results:");
    println!("  Optimal parameters: {:?}", best_result.optimal_params);
    println!("  Ground state energy: {:.8}", best_result.optimal_value);
    println!("  Theoretical minimum: {:.8}", -1.5); // -3J/2 for this Hamiltonian
    println!("  Error: {:.2e}", (best_result.optimal_value + 1.5).abs());
    
    // Analyze the final state
    let final_circuit = ansatz(&best_result.optimal_params);
    let final_state = final_circuit.execute()?;
    
    println!("\nFinal quantum state:");
    let basis_states = ["00", "01", "10", "11"];
    for (i, &label) in basis_states.iter().enumerate() {
        let amplitude = final_state[i];
        if amplitude.norm() > 0.01 {
            println!("  |{}⟩: {:.4} (prob: {:.3})", label, amplitude.norm(), amplitude.norm_sqr());
        }
    }
    
    Ok(())
}

fn hydrogen_molecule_vqe() -> quantum_circuit::Result<()> {
    println!("\n🪃 Hydrogen Molecule (H₂) VQE Simulation");
    println!("----------------------------------------");
    
    // Simplified H2 molecule Hamiltonian (STO-3G basis, R = 0.74 Å)
    // This is a 2-qubit representation using Jordan-Wigner transformation
    let h_coeffs = vec![
        (-1.0523732, "II"),
        (0.39793742, "ZI"), 
        (-0.39793742, "IZ"),
        (-0.01128010, "ZZ"),
        (0.18093119, "XX"),
    ];
    
    println!("Simplified H₂ Hamiltonian (2-qubit Jordan-Wigner mapping):");
    for (coeff, pauli_string) in &h_coeffs {
        println!("  {:.8} × {}", coeff, pauli_string);
    }
    
    // Construct full Hamiltonian
    let hamiltonian = construct_pauli_hamiltonian(&h_coeffs)?;
    
    println!("Theoretical ground state energy: -1.137283 Hartree");
    
    // Create VQE optimizer
    let config = OptimizerConfig {
        max_iterations: 200,
        tolerance: 1e-10,
        learning_rate: 0.02,
        verbose: false,
        ..Default::default()
    };
    
    let mut vqe = VQEOptimizer::new(config, hamiltonian.clone());
    
    // UCCSD-inspired ansatz (Unitary Coupled Cluster Singles and Doubles)
    let ansatz = |params: &[f64]| -> Circuit {
        let mut circuit = Circuit::new(2);
        
        // Initial Hartree-Fock state preparation |01⟩ (two electrons, opposite spins)
        circuit.add_gate(Box::new(PauliX::new(1))).unwrap();
        
        // UCCSD excitation
        // Single excitation: 0→1, 1→0
        circuit.add_gate(Box::new(RY::new(0, params[0]))).unwrap();
        circuit.add_gate(Box::new(RY::new(1, -params[0]))).unwrap(); // Preserve particle number
        
        // Add entanglement
        circuit.add_gate(Box::new(CNOT::new(0, 1))).unwrap();
        
        // Double excitation amplitude
        circuit.add_gate(Box::new(RY::new(0, params[1]))).unwrap();
        circuit.add_gate(Box::new(RY::new(1, params[1]))).unwrap();
        
        circuit.add_gate(Box::new(CNOT::new(1, 0))).unwrap();
        
        // Final single-qubit corrections
        circuit.add_gate(Box::new(RZ::new(0, params[2]))).unwrap();
        circuit.add_gate(Box::new(RZ::new(1, params[3]))).unwrap();
        
        circuit
    };
    
    // Cost function
    let cost_function = |circuit: &Circuit, params: &[f64]| -> quantum_circuit::Result<f64> {
        let energy = circuit.expectation_value_with_parameters(&hamiltonian, params)?;
        Ok(energy)
    };
    
    // Start from small parameter values (close to Hartree-Fock)
    let initial_params = vec![0.1, -0.05, 0.0, 0.0];
    
    println!("\nStarting H₂ VQE optimization...");
    println!("Initial parameters: {:?}", initial_params);
    
    let result = vqe.optimize_circuit(&ansatz(&[0.0; 4]), cost_function, &initial_params)?;
    
    println!("\nH₂ VQE Results:");
    println!("  Optimal parameters: {:?}", result.optimal_parameters);
    println!("  Ground state energy: {:.8} Hartree", result.optimal_value);
    println!("  Theoretical energy: {:.8} Hartree", -1.137283);
    println!("  Chemical accuracy (1 kcal/mol = 0.0016 Hartree)");
    
    let error_hartree = (result.optimal_value + 1.137283).abs();
    let error_kcal_mol = error_hartree / 0.0016;
    
    println!("  Energy error: {:.8} Hartree ({:.3} kcal/mol)", error_hartree, error_kcal_mol);
    
    if error_hartree < 0.0016 {
        println!("  ✅ Chemical accuracy achieved!");
    } else {
        println!("  ⚠️  Chemical accuracy not reached (need better ansatz)");
    }
    
    println!("  Iterations: {}", result.iterations);
    println!("  Function evaluations: {}", result.function_evaluations);
    
    // Analyze the final molecular state
    let final_circuit = ansatz(&result.optimal_params);
    let final_state = final_circuit.execute()?;
    
    println!("\nFinal molecular wave function:");
    let basis_states = ["00 (empty)", "01 (HF state)", "10 (excited)", "11 (doubly excited)"];
    let mut max_amplitude = 0.0;
    let mut dominant_state = 0;
    
    for (i, &label) in basis_states.iter().enumerate() {
        let amplitude = final_state[i];
        let probability = amplitude.norm_sqr();
        
        if amplitude.norm() > 0.01 {
            println!("  |{}⟩: {:.4} (prob: {:.3}%)", label, amplitude.norm(), 100.0 * probability);
        }
        
        if amplitude.norm() > max_amplitude {
            max_amplitude = amplitude.norm();
            dominant_state = i;
        }
    }
    
    println!("\nDominant configuration: {}", basis_states[dominant_state]);
    
    // Calculate some molecular properties
    let bond_length = 0.74; // Å
    let total_energy = result.optimal_value; // Hartree
    let nuclear_repulsion = 1.0 / bond_length * 0.5291772; // Convert to Hartree
    let electronic_energy = total_energy - nuclear_repulsion;
    
    println!("\nMolecular Properties:");
    println!("  Bond length: {:.2} Å", bond_length);
    println!("  Total energy: {:.8} Hartree", total_energy);
    println!("  Electronic energy: {:.8} Hartree", electronic_energy);
    println!("  Nuclear repulsion: {:.8} Hartree", nuclear_repulsion);
    
    Ok(())
}

// Helper functions
fn tensor_product(a: &Array2<quantum_circuit::Complex>, b: &Array2<quantum_circuit::Complex>) -> Array2<quantum_circuit::Complex> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    
    let mut result = Array2::zeros((a_rows * b_rows, a_cols * b_cols));
    
    for i in 0..a_rows {
        for j in 0..a_cols {
            for k in 0..b_rows {
                for l in 0..b_cols {
                    result[[i * b_rows + k, j * b_cols + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }
    
    result
}

fn construct_pauli_hamiltonian(terms: &[(f64, &str)]) -> quantum_circuit::Result<Array2<quantum_circuit::Complex>> {
    let mut hamiltonian = Array2::zeros((4, 4)); // 2-qubit system
    
    for &(coeff, pauli_string) in terms {
        let term_matrix = match pauli_string {
            "II" => tensor_product(&constants::identity(), &constants::identity()),
            "ZI" => tensor_product(&constants::pauli_z(), &constants::identity()),
            "IZ" => tensor_product(&constants::identity(), &constants::pauli_z()),
            "ZZ" => tensor_product(&constants::pauli_z(), &constants::pauli_z()),
            "XX" => tensor_product(&constants::pauli_x(), &constants::pauli_x()),
            "YY" => tensor_product(&constants::pauli_y(), &constants::pauli_y()),
            "XY" => tensor_product(&constants::pauli_x(), &constants::pauli_y()),
            "YX" => tensor_product(&constants::pauli_y(), &constants::pauli_x()),
            _ => return Err(quantum_circuit::QuantumError::InvalidParameter(
                format!("Unknown Pauli string: {}", pauli_string)
            )),
        };
        
        hamiltonian = hamiltonian + coeff * term_matrix;
    }
    
    Ok(hamiltonian)
}