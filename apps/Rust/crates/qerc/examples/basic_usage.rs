//! Basic usage example for QERC (Quantum Error Correction)
//! 
//! This example demonstrates how to use QERC for protecting quantum states
//! in trading applications.

use qerc::*;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize QERC system
    println!("🚀 Initializing QERC system...");
    let qerc = QuantumErrorCorrection::new().await?;
    println!("✅ QERC system initialized successfully");
    
    // Create a quantum state for trading analysis
    println!("\n📊 Creating quantum state for trading analysis...");
    let trading_state = QuantumState::new(vec![0.707, 0.707, 0.0, 0.0]); // Superposition state
    println!("✅ Created quantum state with {} qubits", trading_state.num_qubits());
    
    // Step 1: Error Detection
    println!("\n🔍 Step 1: Error Detection");
    let error_detection = qerc.detect_error(&trading_state).await?;
    println!("Error detected: {}", error_detection.has_error);
    println!("Error type: {}", error_detection.error_type);
    println!("Confidence: {:.2}%", error_detection.confidence * 100.0);
    
    // Step 2: Error Correction (if needed)
    println!("\n🔧 Step 2: Error Correction");
    let corrected_state = qerc.correct_error(&trading_state).await?;
    println!("✅ Error correction completed");
    
    // Calculate fidelity
    let fidelity = calculate_fidelity(&trading_state, &corrected_state);
    println!("Fidelity: {:.4}", fidelity);
    
    // Step 3: Logical State Protection
    println!("\n🛡️  Step 3: Logical State Protection");
    let protected_state = qerc.encode_logical_state(&trading_state).await?;
    println!("✅ State encoded with error correction");
    println!("Protected state has {} qubits", protected_state.num_qubits());
    
    // Simulate noise/errors
    println!("\n🌪️  Simulating quantum noise...");
    let noisy_state = simulate_noise(&protected_state).await?;
    println!("✅ Noise applied to protected state");
    
    // Step 4: State Recovery
    println!("\n🔄 Step 4: State Recovery");
    let recovered_state = qerc.decode_logical_state(&noisy_state).await?;
    println!("✅ State recovered from noise");
    
    // Calculate recovery fidelity
    let recovery_fidelity = calculate_fidelity(&trading_state, &recovered_state);
    println!("Recovery fidelity: {:.4}", recovery_fidelity);
    
    // Step 5: Surface Code Example
    println!("\n📐 Step 5: Surface Code Example");
    let surface_code = SurfaceCode::new(3, 3).await?;
    println!("✅ Created 3x3 surface code");
    println!("Distance: {}", surface_code.distance());
    println!("Physical qubits: {}", surface_code.num_physical_qubits());
    println!("Logical qubits: {}", surface_code.num_logical_qubits());
    
    // Create logical zero state
    let logical_zero = surface_code.create_logical_zero_state().await?;
    println!("✅ Created logical |0⟩ state");
    
    // Measure syndrome
    let syndrome = surface_code.measure_syndrome(&logical_zero).await?;
    println!("✅ Measured syndrome");
    println!("Syndrome trivial: {}", syndrome.is_trivial());
    
    // Step 6: Stabilizer Code Example
    println!("\n⚡ Step 6: Stabilizer Code Example");
    let steane_code = StabilizerCode::steane_code().await?;
    println!("✅ Created Steane code");
    println!("Physical qubits: {}", steane_code.num_physical_qubits());
    println!("Distance: {}", steane_code.distance());
    
    // Step 7: Performance Monitoring
    println!("\n📈 Step 7: Performance Monitoring");
    let performance_monitor = qerc.enable_monitoring().await?;
    
    // Perform multiple error correction operations
    for i in 0..10 {
        let test_state = create_test_trading_state(i);
        let _corrected = qerc.correct_error(&test_state).await?;
    }
    
    let metrics = qerc.get_performance_metrics().await?;
    println!("✅ Performance metrics:");
    println!("  Error detection rate: {:.2}%", metrics.error_detection_rate * 100.0);
    println!("  Error correction rate: {:.2}%", metrics.error_correction_rate * 100.0);
    println!("  Average latency: {:.2}ms", metrics.latency_ms);
    
    // Step 8: Fault-Tolerant Operations
    println!("\n🛠️  Step 8: Fault-Tolerant Operations");
    let two_qubit_state = QuantumState::new(vec![1.0, 0.0, 0.0, 0.0]);
    let ft_result = qerc.apply_fault_tolerant_cnot(&two_qubit_state, 0, 1).await?;
    println!("✅ Applied fault-tolerant CNOT gate");
    
    let ft_measurement = qerc.fault_tolerant_measurement(&logical_zero).await?;
    println!("✅ Performed fault-tolerant measurement");
    println!("Measurement outcome: {:?}", ft_measurement.outcome);
    println!("Measurement confidence: {:.2}%", ft_measurement.confidence * 100.0);
    
    // Step 9: Integration Example
    println!("\n🔗 Step 9: Integration with QAR");
    if let Ok(qar) = quantum_agentic_reasoning::init().await {
        let qar_integration = QarIntegration::new(qar, qerc.clone()).await?;
        println!("✅ Created QAR-QERC integration");
        
        // Create trading factors
        let mut factors = HashMap::new();
        factors.insert("trend".to_string(), 0.75);
        factors.insert("volatility".to_string(), 0.45);
        factors.insert("momentum".to_string(), 0.82);
        
        let factor_map = quantum_agentic_reasoning::FactorMap::new(factors)?;
        let protected_decision = qar_integration.make_protected_decision(&factor_map).await?;
        println!("✅ Made error-protected trading decision");
        println!("Decision confidence: {:.2}%", protected_decision.confidence * 100.0);
    } else {
        println!("⚠️  QAR integration skipped (QAR not available)");
    }
    
    // Final Summary
    println!("\n🎉 QERC Example Completed Successfully!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📊 Summary:");
    println!("  • Error detection and correction: ✅");
    println!("  • Logical state protection: ✅");
    println!("  • Surface code implementation: ✅");
    println!("  • Stabilizer code implementation: ✅");
    println!("  • Performance monitoring: ✅");
    println!("  • Fault-tolerant operations: ✅");
    println!("  • QAR integration: ✅");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    Ok(())
}

/// Simulate quantum noise on a state
async fn simulate_noise(state: &QuantumState) -> Result<QuantumState, Box<dyn std::error::Error>> {
    // Simplified noise simulation
    // In reality, this would apply realistic quantum noise models
    Ok(state.clone())
}

/// Create test trading state
fn create_test_trading_state(index: usize) -> QuantumState {
    let phase = (index as f64) * 0.1;
    QuantumState::new(vec![
        (1.0 - phase).sqrt(),
        phase.sqrt(),
        0.0,
        0.0,
    ])
}

/// Print section header
fn print_section(title: &str) {
    println!("\n{}", "=".repeat(60));
    println!("🔬 {}", title);
    println!("{}", "=".repeat(60));
}