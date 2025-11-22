//! PennyLane-compatible API demonstration
//!
//! This example shows how to use the PennyLane-compatible API
//! for quantum circuit construction and execution.

use quantum_circuit::{
    pennylane_compat::{device, QNodeBuilder, ParameterShiftGradient},
    optimization::{AdamOptimizer, OptimizerConfig, VariationalOptimizer},
    constants,
};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🍃 PennyLane-Compatible API Demonstration");
    println!("=========================================");
    
    // Example 1: Basic QNode usage
    basic_qnode_example()?;
    
    // Example 2: Parameterized quantum circuits
    parameterized_qnode_example()?;
    
    // Example 3: Quantum machine learning with gradients
    quantum_ml_example()?;
    
    // Example 4: Variational quantum classifier
    variational_classifier_example()?;
    
    Ok(())
}

fn basic_qnode_example() -> quantum_circuit::Result<()> {
    println!("\n🔧 Basic QNode Example");
    println!("----------------------");
    
    // Create a quantum device
    let mut device = device("default.qubit", 2)?;
    
    // Build a simple quantum circuit using the PennyLane-style API
    let mut builder = QNodeBuilder::new(2);
    
    builder
        .hadamard(0)?                    // H|0⟩ on qubit 0
        .cnot(0, 1)?                     // CNOT(0,1)
        .ry(PI / 4.0, 1)?               // RY(π/4) on qubit 1
        .expectation(                    // Measure ⟨Z₀⟩
            constants::pauli_z(), 
            "Z0".to_string()
        )?;
    
    let qnode = builder.build();
    
    println!("Circuit structure:");
    println!("  Qubits: {}", qnode.num_qubits);
    println!("  Operations: {}", qnode.operations.len());
    println!("  Observables: {}", qnode.observables.len());
    
    // Execute the QNode
    let result = qnode.execute(device.as_mut(), None)?;
    
    println!("\nExecution results:");
    println!("  Execution time: {:.2} ms", result.metadata.execution_time_ms);
    println!("  Circuit depth: {}", result.metadata.circuit_depth);
    println!("  Gate count: {}", result.metadata.gate_count);
    
    // Show the final quantum state
    if let Some(state) = &result.state {
        println!("\nFinal quantum state:");
        let basis_states = ["00", "01", "10", "11"];
        for (i, &label) in basis_states.iter().enumerate() {
            let amplitude = state[i];
            if amplitude.norm() > 0.01 {
                println!("  |{}⟩: {:.4} (prob: {:.3})", 
                         label, amplitude.norm(), amplitude.norm_sqr());
            }
        }
    }
    
    // Show expectation values
    println!("\nExpectation values:");
    for (obs_name, value) in &result.expectations {
        println!("  ⟨{}⟩ = {:.6}", obs_name, value);
    }
    
    Ok(())
}

fn parameterized_qnode_example() -> quantum_circuit::Result<()> {
    println!("\n⚙️  Parameterized QNode Example");
    println!("------------------------------");
    
    let mut device = device("default.qubit", 2)?;
    
    // Create a parameterized quantum circuit
    let build_qnode = |params: &[f64]| -> quantum_circuit::Result<_> {
        let mut builder = QNodeBuilder::new(2);
        
        builder
            .ry(params[0], 0)?               // RY(θ₁) on qubit 0
            .rx(params[1], 1)?               // RX(θ₂) on qubit 1  
            .cnot(0, 1)?                     // Entangling gate
            .rz(params[2], 0)?               // RZ(θ₃) on qubit 0
            .expectation(                    // Measure ⟨Z₀ ⊗ Z₁⟩
                tensor_product_zz()?, 
                "ZZ".to_string()
            )?;
            
        Ok(builder.build())
    };
    
    // Test with different parameter values
    let parameter_sets = vec![
        ("Zero params", vec![0.0, 0.0, 0.0]),
        ("Random params", vec![PI/3.0, PI/6.0, PI/4.0]),
        ("Max entanglement", vec![PI/2.0, PI/2.0, 0.0]),
        ("Anti-correlated", vec![PI, PI, PI]),
    ];
    
    println!("Testing different parameter values:");
    
    for (description, params) in parameter_sets {
        println!("\n  {}: {:?}", description, params);
        
        let qnode = build_qnode(&params)?;
        let result = qnode.execute(device.as_mut(), Some(&params))?;
        
        for (obs_name, value) in &result.expectations {
            println!("    ⟨{}⟩ = {:.6}", obs_name, value);
        }
        
        // Show dominant basis states
        if let Some(state) = &result.state {
            let mut state_probs: Vec<(usize, f64)> = state.iter()
                .enumerate()
                .map(|(i, c)| (i, c.norm_sqr()))
                .filter(|(_, prob)| *prob > 0.01)
                .collect();
            
            state_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            if !state_probs.is_empty() {
                let basis_states = ["00", "01", "10", "11"];
                print!("    Dominant states: ");
                for (i, (state_idx, prob)) in state_probs.iter().take(2).enumerate() {
                    if i > 0 { print!(", "); }
                    print!("|{}⟩ ({:.1}%)", basis_states[*state_idx], 100.0 * prob);
                }
                println!();
            }
        }
    }
    
    Ok(())
}

fn quantum_ml_example() -> quantum_circuit::Result<()> {
    println!("\n🧠 Quantum Machine Learning with Gradients");
    println!("-------------------------------------------");
    
    // Define a variational quantum circuit for classification
    let create_classifier = |params: &[f64]| -> quantum_circuit::Result<_> {
        let mut builder = QNodeBuilder::new(2);
        
        // Data encoding (fixed for this example)
        builder
            .ry(PI / 4.0, 0)?
            .rx(PI / 3.0, 1)?
            
            // Variational layer 1
            .ry(params[0], 0)?
            .ry(params[1], 1)?
            .cnot(0, 1)?
            
            // Variational layer 2  
            .rx(params[2], 0)?
            .rx(params[3], 1)?
            .cnot(1, 0)?
            
            // Final rotations
            .rz(params[4], 0)?
            .rz(params[5], 1)?
            
            // Classification measurement
            .expectation(
                constants::pauli_z(),
                "classification".to_string()
            )?;
            
        Ok(builder.build())
    };
    
    let mut device = device("default.qubit", 2)?;
    
    // Define cost function (simplified - normally would use training data)
    let cost_function = |params: &[f64]| -> f64 {
        let qnode = match create_classifier(params) {
            Ok(q) => q,
            Err(_) => return f64::INFINITY,
        };
        
        let result = match qnode.execute(device.as_mut(), Some(params)) {
            Ok(r) => r,
            Err(_) => return f64::INFINITY,
        };
        
        // Target expectation value (example: we want ⟨Z⟩ = 0.5)
        let target = 0.5;
        let prediction = result.expectations.get("classification").unwrap_or(&0.0);
        
        (prediction - target).powi(2) // Mean squared error
    };
    
    // Compute gradients using parameter-shift rule
    let gradient_computer = ParameterShiftGradient::new();
    let initial_params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    
    println!("Initial parameters: {:?}", initial_params);
    println!("Initial cost: {:.6}", cost_function(&initial_params));
    
    // Compute gradients
    let gradients = gradient_computer.compute_gradients(
        |params| Ok(cost_function(params)),
        &initial_params,
        &constants::pauli_z(),
    )?;
    
    println!("Parameter gradients: {:?}", 
             gradients.iter().map(|&g| format!("{:.4}", g)).collect::<Vec<_>>());
    
    // Simple gradient descent step
    let learning_rate = 0.1;
    let updated_params: Vec<f64> = initial_params.iter()
        .zip(gradients.iter())
        .map(|(&param, &grad)| param - learning_rate * grad)
        .collect();
    
    println!("Updated parameters: {:?}", 
             updated_params.iter().map(|&p| format!("{:.3}", p)).collect::<Vec<_>>());
    println!("Updated cost: {:.6}", cost_function(&updated_params));
    
    // Show improvement
    let initial_cost = cost_function(&initial_params);
    let updated_cost = cost_function(&updated_params);
    let improvement = ((initial_cost - updated_cost) / initial_cost) * 100.0;
    
    if improvement > 0.0 {
        println!("✅ Cost improved by {:.2}%", improvement);
    } else {
        println!("⚠️ Cost increased by {:.2}%", -improvement);
    }
    
    Ok(())
}

fn variational_classifier_example() -> quantum_circuit::Result<()> {
    println!("\n🎯 Variational Quantum Classifier");
    println!("----------------------------------");
    
    // Simulate a simple binary classification problem
    // We'll create a parameterized quantum circuit that learns to distinguish
    // between two different input encodings
    
    let mut device = device("default.qubit", 3)?; // Use 3 qubits for more complexity
    
    // Define the variational quantum classifier
    let create_vqc = |data: &[f64], params: &[f64]| -> quantum_circuit::Result<_> {
        let mut builder = QNodeBuilder::new(3);
        
        // Data encoding layer
        builder
            .ry(data[0] * PI, 0)?
            .ry(data[1] * PI, 1)?
            .ry((data[0] + data[1]) * PI / 2.0, 2)?
            
            // Variational ansatz layer 1
            .ry(params[0], 0)?
            .ry(params[1], 1)?
            .ry(params[2], 2)?
            
            // Entangling layer 1
            .cnot(0, 1)?
            .cnot(1, 2)?
            .cnot(2, 0)?
            
            // Variational ansatz layer 2
            .rx(params[3], 0)?
            .rx(params[4], 1)?
            .rx(params[5], 2)?
            
            // Entangling layer 2
            .cnot(0, 2)?
            .cnot(1, 0)?
            
            // Final measurements
            .expectation(
                constants::pauli_z(),
                "qubit_0".to_string()
            )?
            .expectation(
                tensor_product_zi()?,
                "qubit_1".to_string()
            )?;
            
        Ok(builder.build())
    };
    
    // Training dataset (simplified)
    let training_data = vec![
        (vec![0.2, 0.3], 0), // Class 0
        (vec![0.1, 0.4], 0), // Class 0  
        (vec![0.8, 0.7], 1), // Class 1
        (vec![0.9, 0.6], 1), // Class 1
    ];
    
    println!("Training dataset:");
    for (i, (data, label)) in training_data.iter().enumerate() {
        println!("  Sample {}: {:?} -> Class {}", i + 1, data, label);
    }
    
    // Simple training simulation
    let mut params = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    println!("\nInitial parameters: {:?}", 
             params.iter().map(|&p| format!("{:.2}", p)).collect::<Vec<_>>());
    
    // Evaluate initial performance
    println!("\nInitial predictions:");
    for (i, (data, true_label)) in training_data.iter().enumerate() {
        let qnode = create_vqc(data, &params)?;
        let result = qnode.execute(device.as_mut(), Some(&params))?;
        
        let prediction_0 = result.expectations.get("qubit_0").unwrap_or(&0.0);
        let prediction_1 = result.expectations.get("qubit_1").unwrap_or(&0.0);
        
        // Simple classification rule: if prediction_0 > 0, class 0, else class 1
        let predicted_class = if prediction_0 > &0.0 { 0 } else { 1 };
        let confidence = prediction_0.abs();
        
        println!("  Sample {}: Predicted {} (confidence: {:.3}) | True: {} | {}",
                 i + 1, predicted_class, confidence, true_label,
                 if predicted_class == *true_label { "✓" } else { "✗" });
    }
    
    // Simulate parameter optimization (simplified)
    println!("\nSimulating training optimization...");
    
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut best_params = params.clone();
    let mut best_accuracy = 0.0;
    
    for epoch in 0..50 {
        // Random parameter perturbation (simplified optimization)
        let mut new_params = params.clone();
        for param in &mut new_params {
            *param += rng.gen_range(-0.1..0.1);
        }
        
        // Evaluate accuracy
        let mut correct = 0;
        for (data, true_label) in &training_data {
            let qnode = create_vqc(data, &new_params)?;
            let result = qnode.execute(device.as_mut(), Some(&new_params))?;
            
            let prediction_0 = result.expectations.get("qubit_0").unwrap_or(&0.0);
            let predicted_class = if prediction_0 > &0.0 { 0 } else { 1 };
            
            if predicted_class == *true_label {
                correct += 1;
            }
        }
        
        let accuracy = correct as f64 / training_data.len() as f64;
        
        if accuracy > best_accuracy {
            best_accuracy = accuracy;
            best_params = new_params;
            
            if epoch % 10 == 0 {
                println!("  Epoch {}: New best accuracy = {:.2}% | Parameters: {:?}",
                         epoch, 100.0 * accuracy,
                         best_params.iter().map(|&p| format!("{:.2}", p)).collect::<Vec<_>>());
            }
        }
    }
    
    println!("\nFinal trained classifier performance:");
    println!("  Best accuracy: {:.1}%", 100.0 * best_accuracy);
    
    // Test the final model
    println!("\nFinal predictions:");
    for (i, (data, true_label)) in training_data.iter().enumerate() {
        let qnode = create_vqc(data, &best_params)?;
        let result = qnode.execute(device.as_mut(), Some(&best_params))?;
        
        let prediction_0 = result.expectations.get("qubit_0").unwrap_or(&0.0);
        let prediction_1 = result.expectations.get("qubit_1").unwrap_or(&0.0);
        let predicted_class = if prediction_0 > &0.0 { 0 } else { 1 };
        let confidence = prediction_0.abs();
        
        println!("  Sample {}: Predicted {} (conf: {:.3}) | True: {} | {} (⟨Z₀⟩={:.3}, ⟨Z₁⟩={:.3})",
                 i + 1, predicted_class, confidence, true_label,
                 if predicted_class == *true_label { "✓" } else { "✗" },
                 prediction_0, prediction_1);
    }
    
    // Test on new data
    let test_data = vec![
        vec![0.15, 0.35], // Should be class 0
        vec![0.85, 0.75], // Should be class 1
    ];
    
    println!("\nTesting on new data:");
    for (i, data) in test_data.iter().enumerate() {
        let qnode = create_vqc(data, &best_params)?;
        let result = qnode.execute(device.as_mut(), Some(&best_params))?;
        
        let prediction_0 = result.expectations.get("qubit_0").unwrap_or(&0.0);
        let predicted_class = if prediction_0 > &0.0 { 0 } else { 1 };
        let confidence = prediction_0.abs();
        
        println!("  Test sample {}: {:?} -> Class {} (confidence: {:.3})",
                 i + 1, data, predicted_class, confidence);
    }
    
    Ok(())
}

// Helper functions
fn tensor_product_zz() -> quantum_circuit::Result<ndarray::Array2<quantum_circuit::Complex>> {
    let z = constants::pauli_z();
    let id = constants::identity();
    
    // Z ⊗ Z for 2-qubit system
    Ok(tensor_product(&z, &z))
}

fn tensor_product_zi() -> quantum_circuit::Result<ndarray::Array2<quantum_circuit::Complex>> {
    let z = constants::pauli_z();
    let id = constants::identity();
    
    // Z ⊗ I for measuring just the second qubit
    Ok(tensor_product(&z, &id))
}

fn tensor_product(
    a: &ndarray::Array2<quantum_circuit::Complex>, 
    b: &ndarray::Array2<quantum_circuit::Complex>
) -> ndarray::Array2<quantum_circuit::Complex> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    
    let mut result = ndarray::Array2::zeros((a_rows * b_rows, a_cols * b_cols));
    
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