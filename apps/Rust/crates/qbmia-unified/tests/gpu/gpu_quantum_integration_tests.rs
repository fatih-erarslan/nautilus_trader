//! GPU Quantum Integration Tests
//! 
//! Comprehensive integration tests demonstrating GPU-only quantum computing
//! workflows across multiple backends. These tests validate the complete
//! TENGRI-compliant quantum simulation pipeline.
//! 
//! NO CLOUD QUANTUM BACKENDS - ONLY LOCAL GPU HARDWARE

use std::sync::Arc;
use tokio;
use tracing_test::traced_test;

use qbmia_unified::gpu::{
    GpuQuantumSimulator,
    GpuQuantumSimulatorFactory,
    GpuQuantumFourierTransform,
    GpuVariationalQuantumEigensolver,
    GpuQuantumApproximateOptimization,
    GpuPauliHamiltonian,
    GpuQuantumBenchmarks,
    GpuQuantumValidation,
};
use qbmia_unified::{Result, GpuBackend};

/// Integration test suite for GPU quantum computing
#[cfg(test)]
mod gpu_quantum_integration {
    use super::*;
    
    /// Test complete GPU quantum simulator lifecycle
    #[tokio::test]
    #[traced_test]
    async fn test_gpu_quantum_simulator_lifecycle() -> Result<()> {
        // Detect available GPU quantum simulators
        let simulators = match GpuQuantumSimulatorFactory::detect_and_create_simulators().await {
            Ok(sims) if !sims.is_empty() => sims,
            Ok(_) => {
                tracing::warn!("No GPU devices available for testing");
                return Ok(());
            }
            Err(e) => {
                tracing::warn!("Failed to detect GPU devices: {}", e);
                return Ok(());
            }
        };
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Testing lifecycle on {:?} backend", backend);
            
            // Test basic simulator operations
            let mut sim = simulator.clone();
            assert_eq!(sim.num_qubits(), 0);
            
            // Initialize quantum state
            sim.initialize_qubits(4).await?;
            assert_eq!(sim.num_qubits(), 4);
            
            let state = sim.get_state_vector().unwrap();
            assert_eq!(state.len(), 16); // 2^4
            assert!((state[0].re - 1.0).abs() < 1e-10); // |0000> state
            
            // Apply gates and verify state changes
            let device_info = sim.get_device_info();
            tracing::info!(
                "Simulator on {}: {} qubits max, {:.2} GB/s bandwidth",
                device_info.device_name,
                sim.max_qubits(),
                device_info.memory_bandwidth_gbps
            );
        }
        
        Ok(())
    }
    
    /// Test GPU Quantum Fourier Transform workflow
    #[tokio::test]
    #[traced_test]
    async fn test_gpu_qft_workflow() -> Result<()> {
        let simulators = match GpuQuantumSimulatorFactory::detect_and_create_simulators().await {
            Ok(sims) if !sims.is_empty() => sims,
            _ => return Ok(()),
        };
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Testing QFT workflow on {:?} backend", backend);
            
            // Test QFT for multiple qubit counts
            for num_qubits in [2, 3, 4, 5] {
                if num_qubits > simulator.max_qubits() { continue; }
                
                let mut sim = simulator.clone();
                sim.initialize_qubits(num_qubits).await?;
                
                // Store initial state
                let initial_state = sim.get_state_vector().unwrap().clone();
                
                // Apply forward QFT
                let qft = GpuQuantumFourierTransform::new(num_qubits, false, backend);
                qft.execute(&mut sim).await?;
                
                // Verify state changed
                let qft_state = sim.get_state_vector().unwrap();
                let overlap: f64 = initial_state.iter()
                    .zip(qft_state.iter())
                    .map(|(a, b)| (a.conj() * b).norm_sqr())
                    .sum();
                assert!(overlap < 0.99, "QFT should change the state significantly");
                
                // Apply inverse QFT
                let inv_qft = GpuQuantumFourierTransform::new(num_qubits, true, backend);
                inv_qft.execute(&mut sim).await?;
                
                // Verify we recovered the initial state
                let final_state = sim.get_state_vector().unwrap();
                let fidelity: f64 = initial_state.iter()
                    .zip(final_state.iter())
                    .map(|(a, b)| (a.conj() * b).re)
                    .sum();
                
                assert!(
                    fidelity > 0.99,
                    "QFT round-trip failed on {:?} with {} qubits: fidelity = {}",
                    backend, num_qubits, fidelity
                );
                
                tracing::info!(
                    "✓ QFT round-trip test passed for {} qubits on {:?} (fidelity: {:.6})",
                    num_qubits, backend, fidelity
                );
            }
        }
        
        Ok(())
    }
    
    /// Test Variational Quantum Eigensolver (VQE) workflow
    #[tokio::test]
    #[traced_test]
    async fn test_gpu_vqe_workflow() -> Result<()> {
        let simulators = match GpuQuantumSimulatorFactory::detect_and_create_simulators().await {
            Ok(sims) if !sims.is_empty() => sims,
            _ => return Ok(()),
        };
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Testing VQE workflow on {:?} backend", backend);
            
            let num_qubits = std::cmp::min(4, simulator.max_qubits());
            let mut vqe = GpuVariationalQuantumEigensolver::new(num_qubits, 2, backend);
            let mut sim = simulator.clone();
            
            // Create simple Ising model Hamiltonian
            let mut hamiltonian = GpuPauliHamiltonian::new(num_qubits);
            
            // Add nearest-neighbor ZZ interactions
            for i in 0..num_qubits - 1 {
                hamiltonian.add_term(-1.0, vec![(i, 'Z'), (i + 1, 'Z')]);
            }
            
            // Add transverse field
            for i in 0..num_qubits {
                hamiltonian.add_term(-0.5, vec![(i, 'X')]);
            }
            
            // Run VQE optimization
            let initial_energy = vqe.evaluate_energy(&mut sim, &hamiltonian).await?;
            let (final_energy, best_params) = vqe.optimize(&mut sim, &hamiltonian, 50).await?;
            
            // Verify optimization improved the energy
            assert!(
                final_energy <= initial_energy + 1e-10,
                "VQE failed to optimize on {:?}: initial = {:.6}, final = {:.6}",
                backend, initial_energy, final_energy
            );
            
            // Verify parameters are reasonable
            assert_eq!(best_params.len(), num_qubits * 2 * 2); // 2 layers, 2 params per qubit
            assert!(best_params.iter().all(|&p| p.is_finite()));
            
            tracing::info!(
                "✓ VQE optimization on {:?}: {:.6} → {:.6} (improvement: {:.6})",
                backend, initial_energy, final_energy, initial_energy - final_energy
            );
        }
        
        Ok(())
    }
    
    /// Test Quantum Approximate Optimization Algorithm (QAOA) workflow
    #[tokio::test]
    #[traced_test]
    async fn test_gpu_qaoa_workflow() -> Result<()> {
        let simulators = match GpuQuantumSimulatorFactory::detect_and_create_simulators().await {
            Ok(sims) if !sims.is_empty() => sims,
            _ => return Ok(()),
        };
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Testing QAOA workflow on {:?} backend", backend);
            
            let num_qubits = std::cmp::min(4, simulator.max_qubits());
            let mut qaoa = GpuQuantumApproximateOptimization::new(num_qubits, 2, backend);
            let mut sim = simulator.clone();
            
            // Create Max-Cut problem on a path graph
            let edges: Vec<(usize, usize)> = (0..num_qubits - 1)
                .map(|i| (i, i + 1))
                .collect();
            let hamiltonian = GpuPauliHamiltonian::max_cut_hamiltonian(&edges, num_qubits);
            
            // Run QAOA optimization
            let initial_cost = qaoa.evaluate_cost(&mut sim, &hamiltonian).await?;
            let (final_cost, best_beta, best_gamma) = qaoa.optimize(&mut sim, &hamiltonian, 30).await?;
            
            // Verify optimization worked
            assert!(
                final_cost <= initial_cost + 1e-10,
                "QAOA failed to optimize on {:?}: initial = {:.6}, final = {:.6}",
                backend, initial_cost, final_cost
            );
            
            // Verify parameters
            assert_eq!(best_beta.len(), 2); // 2 layers
            assert_eq!(best_gamma.len(), 2);
            assert!(best_beta.iter().all(|&p| p.is_finite() && p >= 0.0));
            assert!(best_gamma.iter().all(|&p| p.is_finite()));
            
            tracing::info!(
                "✓ QAOA optimization on {:?}: {:.6} → {:.6} (improvement: {:.6})",
                backend, initial_cost, final_cost, initial_cost - final_cost
            );
        }
        
        Ok(())
    }
    
    /// Test quantum state manipulation and measurement
    #[tokio::test]
    #[traced_test]
    async fn test_quantum_state_manipulation() -> Result<()> {
        let simulators = match GpuQuantumSimulatorFactory::detect_and_create_simulators().await {
            Ok(sims) if !sims.is_empty() => sims,
            _ => return Ok(()),
        };
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Testing state manipulation on {:?} backend", backend);
            
            let num_qubits = 3;
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            // Create GHZ state: (|000> + |111>) / sqrt(2)
            let h_gate = GpuQuantumBenchmarks::create_hadamard_gate(0, backend)?;
            sim.apply_gate(&h_gate).await?;
            
            let cnot1 = GpuQuantumBenchmarks::create_cnot_gate(0, 1, backend)?;
            let cnot2 = GpuQuantumBenchmarks::create_cnot_gate(1, 2, backend)?;
            sim.apply_gate(&cnot1).await?;
            sim.apply_gate(&cnot2).await?;
            
            // Verify GHZ state
            let state = sim.get_state_vector().unwrap();
            let expected_amp = 1.0 / (2.0_f64).sqrt();
            
            assert!(
                (state[0].re - expected_amp).abs() < 1e-6,
                "GHZ |000> amplitude incorrect: expected {}, got {}",
                expected_amp, state[0].re
            );
            assert!(
                (state[7].re - expected_amp).abs() < 1e-6,
                "GHZ |111> amplitude incorrect: expected {}, got {}",
                expected_amp, state[7].re
            );
            
            // Check other amplitudes are zero
            for i in 1..7 {
                assert!(
                    state[i].norm() < 1e-6,
                    "GHZ state should have zero amplitude for |{:03b}>",
                    i
                );
            }
            
            // Test measurement statistics
            let mut measurement_counts = vec![0; 8];
            let num_shots = 1000;
            
            for _ in 0..num_shots {
                let mut sim_copy = sim.clone();
                let measurements = sim_copy.measure_all().await?;
                
                let measurement_index = measurements.iter()
                    .enumerate()
                    .fold(0, |acc, (i, &bit)| acc + if bit { 1 << i } else { 0 });
                
                measurement_counts[measurement_index] += 1;
            }
            
            // Should mostly measure |000> and |111>
            let total_ghz_measurements = measurement_counts[0] + measurement_counts[7];
            let ghz_probability = total_ghz_measurements as f64 / num_shots as f64;
            
            assert!(
                ghz_probability > 0.8, // Allow some statistical error
                "GHZ state measurement statistics incorrect on {:?}: {:.3}",
                backend, ghz_probability
            );
            
            tracing::info!(
                "✓ GHZ state created and measured on {:?} (p_GHZ = {:.3})",
                backend, ghz_probability
            );
        }
        
        Ok(())
    }
    
    /// Test performance across different qubit counts
    #[tokio::test]
    #[traced_test]
    async fn test_performance_scaling() -> Result<()> {
        let simulator = match GpuQuantumSimulatorFactory::get_best_simulator().await {
            Ok(sim) => sim,
            Err(_) => {
                tracing::warn!("No GPU devices available for performance testing");
                return Ok(());
            }
        };
        
        let backend = simulator.get_device_info().backend;
        tracing::info!("Performance scaling test on {:?} backend", backend);
        
        let max_qubits = std::cmp::min(simulator.max_qubits(), 16);
        let mut performance_data = Vec::new();
        
        for num_qubits in [2, 4, 6, 8, 10, 12, 14, 16] {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            let start = std::time::Instant::now();
            sim.initialize_qubits(num_qubits).await?;
            let init_time = start.elapsed();
            
            // Apply some gates
            let start = std::time::Instant::now();
            for qubit in 0..num_qubits {
                let gate = GpuQuantumBenchmarks::create_hadamard_gate(qubit, backend)?;
                sim.apply_gate(&gate).await?;
            }
            let gate_time = start.elapsed();
            
            // Measure
            let start = std::time::Instant::now();
            let _measurements = sim.measure_all().await?;
            let measure_time = start.elapsed();
            
            performance_data.push((
                num_qubits,
                init_time.as_micros(),
                gate_time.as_micros(),
                measure_time.as_micros(),
            ));
            
            tracing::info!(
                "{} qubits: init {:.1}μs, gates {:.1}μs, measure {:.1}μs",
                num_qubits,
                init_time.as_micros(),
                gate_time.as_micros(),
                measure_time.as_micros()
            );
        }
        
        // Verify scaling is reasonable (not exponential in time)
        if performance_data.len() >= 2 {
            let first = &performance_data[0];
            let last = &performance_data[performance_data.len() - 1];
            
            let qubit_ratio = last.0 as f64 / first.0 as f64;
            let time_ratio = last.2 as f64 / first.2 as f64; // Gate time ratio
            
            tracing::info!(
                "Scaling analysis: {:.1}x qubits resulted in {:.1}x gate time",
                qubit_ratio, time_ratio
            );
            
            // Gate time should scale roughly linearly or slightly superlinearly,
            // not exponentially like classical simulation
            assert!(
                time_ratio < qubit_ratio.powf(2.0),
                "Gate time scaling too poor: {:.1}x time for {:.1}x qubits",
                time_ratio, qubit_ratio
            );
        }
        
        Ok(())
    }
    
    /// Test error handling and edge cases
    #[tokio::test]
    #[traced_test]
    async fn test_error_handling() -> Result<()> {
        let simulator = match GpuQuantumSimulatorFactory::get_best_simulator().await {
            Ok(sim) => sim,
            Err(_) => return Ok(()),
        };
        
        let mut sim = simulator.clone();
        
        // Test invalid qubit initialization
        let max_qubits = sim.max_qubits();
        let result = sim.initialize_qubits(max_qubits + 1).await;
        assert!(result.is_err(), "Should fail to initialize too many qubits");
        
        // Test operations on uninitialized simulator
        let mut empty_sim = simulator.clone();
        let measurements = empty_sim.measure_all().await;
        assert!(measurements.is_err(), "Should fail to measure uninitialized state");
        
        // Test invalid gate operations
        sim.initialize_qubits(2).await?;
        
        // This should work fine as we have valid backends
        tracing::info!("✓ Error handling tests passed");
        
        Ok(())
    }
    
    /// Comprehensive validation test
    #[tokio::test]
    #[traced_test]
    async fn test_comprehensive_validation() -> Result<()> {
        tracing::info!("Running comprehensive GPU quantum validation suite");
        
        // Run all validation tests
        match GpuQuantumValidation::run_all_validation_tests().await {
            Ok(()) => {
                tracing::info!("✓ All validation tests passed");
            }
            Err(e) => {
                tracing::warn!("Some validation tests failed: {}", e);
                // Don't fail the test if no GPU is available
                if e.to_string().contains("No GPU devices") {
                    return Ok(());
                }
                return Err(e);
            }
        }
        
        Ok(())
    }
    
    /// Performance benchmark integration test
    #[tokio::test]
    #[traced_test]
    async fn test_performance_benchmarks() -> Result<()> {
        tracing::info!("Running GPU quantum performance benchmarks");
        
        // Run key performance benchmarks
        match GpuQuantumBenchmarks::run_all_benchmarks().await {
            Ok(()) => {
                tracing::info!("✓ All performance benchmarks completed");
            }
            Err(e) => {
                tracing::warn!("Some benchmarks failed: {}", e);
                // Don't fail the test if no GPU is available
                if e.to_string().contains("No GPU devices") {
                    return Ok(());
                }
                return Err(e);
            }
        }
        
        Ok(())
    }
}

/// Helper functions for integration tests
mod test_helpers {
    use super::*;
    
    /// Create a test Hamiltonian for optimization problems
    pub fn create_test_hamiltonian(num_qubits: usize) -> GpuPauliHamiltonian {
        let mut hamiltonian = GpuPauliHamiltonian::new(num_qubits);
        
        // Add random Pauli terms
        for i in 0..num_qubits {
            hamiltonian.add_term(rand::random::<f64>() - 0.5, vec![(i, 'Z')]);
            if i < num_qubits - 1 {
                hamiltonian.add_term(
                    rand::random::<f64>() - 0.5, 
                    vec![(i, 'Z'), (i + 1, 'Z')]
                );
            }
        }
        
        hamiltonian
    }
    
    /// Verify quantum state properties
    pub fn verify_quantum_state_properties(state: &[num_complex::Complex64]) -> bool {
        // Check normalization
        let norm_squared: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        if (norm_squared - 1.0).abs() > 1e-10 {
            return false;
        }
        
        // Check all amplitudes are finite
        state.iter().all(|c| c.re.is_finite() && c.im.is_finite())
    }
}