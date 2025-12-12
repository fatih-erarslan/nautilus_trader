//! GPU Quantum Computing Performance Benchmarks
//! 
//! Comprehensive benchmarks for GPU-accelerated quantum algorithms.
//! Tests performance across CUDA, OpenCL, Vulkan, and Metal backends.
//! TENGRI COMPLIANT - ALL BENCHMARKS USE REAL GPU HARDWARE.

use std::sync::Arc;
use std::time::Instant;
use criterion::{BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use ndarray::Array1;

use super::{
    GpuQuantumSimulator, 
    GpuQuantumSimulatorFactory,
    GpuQuantumFourierTransform,
    GpuVariationalQuantumEigensolver,
    GpuQuantumApproximateOptimization,
    GpuPauliHamiltonian,
};
use crate::{Result, GpuBackend};

/// GPU quantum benchmark suite
pub struct GpuQuantumBenchmarks;

impl GpuQuantumBenchmarks {
    /// Run all GPU quantum benchmarks
    pub async fn run_all_benchmarks() -> Result<()> {
        let simulators = GpuQuantumSimulatorFactory::detect_and_create_simulators().await?;
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Running benchmarks on {} backend", backend);
            
            Self::benchmark_single_qubit_gates(&simulator).await?;
            Self::benchmark_two_qubit_gates(&simulator).await?;
            Self::benchmark_qft_performance(&simulator).await?;
            Self::benchmark_vqe_performance(&simulator).await?;
            Self::benchmark_qaoa_performance(&simulator).await?;
            Self::benchmark_state_preparation(&simulator).await?;
            Self::benchmark_measurement(&simulator).await?;
        }
        
        Ok(())
    }
    
    /// Benchmark single-qubit gate performance
    async fn benchmark_single_qubit_gates(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 20);
        
        for num_qubits in [1, 2, 4, 8, 16].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            // Benchmark Hadamard gates
            let start = Instant::now();
            for qubit in 0..num_qubits {
                let gate = Self::create_hadamard_gate(qubit, backend)?;
                sim.apply_gate(&gate).await?;
            }
            let duration = start.elapsed();
            
            let gates_per_second = num_qubits as f64 / duration.as_secs_f64();
            tracing::info!(
                "{:?} backend: {} Hadamard gates on {} qubits: {:.0} gates/sec",
                backend, num_qubits, num_qubits, gates_per_second
            );
        }
        
        Ok(())
    }
    
    /// Benchmark two-qubit gate performance
    async fn benchmark_two_qubit_gates(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 20);
        
        for num_qubits in [2, 4, 8, 16].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            // Benchmark CNOT gates
            let start = Instant::now();
            for qubit in 0..num_qubits - 1 {
                let gate = Self::create_cnot_gate(qubit, qubit + 1, backend)?;
                sim.apply_gate(&gate).await?;
            }
            let duration = start.elapsed();
            
            let gates_per_second = (num_qubits - 1) as f64 / duration.as_secs_f64();
            tracing::info!(
                "{:?} backend: {} CNOT gates on {} qubits: {:.0} gates/sec",
                backend, num_qubits - 1, num_qubits, gates_per_second
            );
        }
        
        Ok(())
    }
    
    /// Benchmark QFT performance across different qubit counts
    async fn benchmark_qft_performance(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 16);
        
        for num_qubits in [2, 4, 6, 8, 10, 12, 14, 16].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            // Benchmark forward QFT
            let qft = GpuQuantumFourierTransform::new(num_qubits, false, backend);
            let start = Instant::now();
            qft.execute(&mut sim).await?;
            let duration = start.elapsed();
            
            tracing::info!(
                "{:?} backend: QFT on {} qubits: {:.2} ms ({:.0} qubits/sec)",
                backend, num_qubits, duration.as_millis(), 
                num_qubits as f64 / duration.as_secs_f64()
            );
            
            // Verify inverse QFT returns to original state
            let inv_qft = GpuQuantumFourierTransform::new(num_qubits, true, backend);
            inv_qft.execute(&mut sim).await?;
            
            // Should be back to |00...0> state
            let state = sim.get_state_vector().unwrap();
            let fidelity = state[0].norm_sqr();
            assert!(fidelity > 0.99, "Inverse QFT failed: fidelity = {}", fidelity);
        }
        
        Ok(())
    }
    
    /// Benchmark VQE performance
    async fn benchmark_vqe_performance(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 8);
        
        for num_qubits in [2, 4, 6, 8].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut vqe = GpuVariationalQuantumEigensolver::new(num_qubits, 2, backend);
            let mut sim = simulator.clone();
            
            // Create simple Hamiltonian (Ising model)
            let mut hamiltonian = GpuPauliHamiltonian::new(num_qubits);
            for i in 0..num_qubits - 1 {
                hamiltonian.add_term(-1.0, vec![(i, 'Z'), (i + 1, 'Z')]);
            }
            for i in 0..num_qubits {
                hamiltonian.add_term(-0.5, vec![(i, 'X')]);
            }
            
            let start = Instant::now();
            let (energy, _) = vqe.optimize(&mut sim, &hamiltonian, 20).await?;
            let duration = start.elapsed();
            
            tracing::info!(
                "{:?} backend: VQE on {} qubits: {:.2} ms, energy = {:.4}",
                backend, num_qubits, duration.as_millis(), energy
            );
        }
        
        Ok(())
    }
    
    /// Benchmark QAOA performance
    async fn benchmark_qaoa_performance(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 8);
        
        for num_qubits in [4, 6, 8].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut qaoa = GpuQuantumApproximateOptimization::new(num_qubits, 2, backend);
            let mut sim = simulator.clone();
            
            // Create Max-Cut Hamiltonian
            let edges: Vec<(usize, usize)> = (0..num_qubits - 1)
                .map(|i| (i, i + 1))
                .collect();
            let hamiltonian = GpuPauliHamiltonian::max_cut_hamiltonian(&edges, num_qubits);
            
            let start = Instant::now();
            let (cost, _, _) = qaoa.optimize(&mut sim, &hamiltonian, 20).await?;
            let duration = start.elapsed();
            
            tracing::info!(
                "{:?} backend: QAOA on {} qubits: {:.2} ms, cost = {:.4}",
                backend, num_qubits, duration.as_millis(), cost
            );
        }
        
        Ok(())
    }
    
    /// Benchmark state preparation performance
    async fn benchmark_state_preparation(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 20);
        
        for num_qubits in [1, 2, 4, 8, 16, 20].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            
            // Benchmark initialization time
            let start = Instant::now();
            sim.initialize_qubits(num_qubits).await?;
            let duration = start.elapsed();
            
            let state_size = 1 << num_qubits;
            let complex_numbers_per_second = state_size as f64 / duration.as_secs_f64();
            
            tracing::info!(
                "{:?} backend: Initialize {} qubits ({} amplitudes): {:.2} μs ({:.0} amplitudes/sec)",
                backend, num_qubits, state_size, duration.as_micros(),
                complex_numbers_per_second
            );
        }
        
        Ok(())
    }
    
    /// Benchmark measurement performance
    async fn benchmark_measurement(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 20);
        
        for num_qubits in [1, 2, 4, 8, 16, 20].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            // Prepare superposition state
            for qubit in 0..num_qubits {
                let gate = Self::create_hadamard_gate(qubit, backend)?;
                sim.apply_gate(&gate).await?;
            }
            
            // Benchmark measurement time
            let start = Instant::now();
            let _measurements = sim.measure_all().await?;
            let duration = start.elapsed();
            
            tracing::info!(
                "{:?} backend: Measure {} qubits: {:.2} μs",
                backend, num_qubits, duration.as_micros()
            );
        }
        
        Ok(())
    }
    
    /// Memory bandwidth benchmark for quantum state vectors
    pub async fn benchmark_memory_bandwidth(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let device_info = simulator.get_device_info();
        
        tracing::info!(
            "Memory bandwidth benchmark for {} (theoretical: {:.2} GB/s)",
            device_info.device_name, device_info.memory_bandwidth_gbps
        );
        
        // Test different qubit counts to measure actual bandwidth
        let max_qubits = std::cmp::min(simulator.max_qubits(), 25);
        
        for num_qubits in [10, 15, 20, 25].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            let state_size = 1 << num_qubits;
            let memory_size = state_size * std::mem::size_of::<Complex64>();
            
            // Apply many gates to stress memory bandwidth
            let start = Instant::now();
            for _ in 0..100 {
                let gate = Self::create_hadamard_gate(0, backend)?;
                sim.apply_gate(&gate).await?;
            }
            let duration = start.elapsed();
            
            let operations = 100;
            let total_memory_accessed = memory_size * operations * 2; // Read + Write
            let bandwidth_gbps = (total_memory_accessed as f64 / duration.as_secs_f64()) 
                / (1024.0 * 1024.0 * 1024.0);
            
            tracing::info!(
                "{:?} backend: {} qubits ({:.1} MB state): {:.2} GB/s effective bandwidth",
                backend, num_qubits, memory_size as f64 / (1024.0 * 1024.0), bandwidth_gbps
            );
        }
        
        Ok(())
    }
    
    /// Performance scaling analysis
    pub async fn analyze_scaling_performance() -> Result<()> {
        let simulators = GpuQuantumSimulatorFactory::detect_and_create_simulators().await?;
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Scaling analysis for {:?} backend", backend);
            
            Self::analyze_gate_scaling(&simulator).await?;
            Self::analyze_qft_scaling(&simulator).await?;
        }
        
        Ok(())
    }
    
    async fn analyze_gate_scaling(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let qubit_counts = vec![2, 4, 6, 8, 10, 12, 14, 16];
        let max_qubits = std::cmp::min(simulator.max_qubits(), 16);
        
        let mut results = Vec::new();
        
        for &num_qubits in &qubit_counts {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            // Apply single Hadamard gate and measure time
            let start = Instant::now();
            let gate = Self::create_hadamard_gate(0, backend)?;
            sim.apply_gate(&gate).await?;
            let duration = start.elapsed();
            
            let state_size = 1 << num_qubits;
            results.push((num_qubits, state_size, duration.as_nanos()));
        }
        
        // Analyze scaling
        tracing::info!("Gate application scaling for {:?}:", backend);
        tracing::info!("Qubits | State Size | Time (ns) | ns/amplitude");
        for (qubits, state_size, time_ns) in results {
            let ns_per_amplitude = time_ns as f64 / state_size as f64;
            tracing::info!(
                "{:6} | {:10} | {:9} | {:12.2}",
                qubits, state_size, time_ns, ns_per_amplitude
            );
        }
        
        Ok(())
    }
    
    async fn analyze_qft_scaling(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let qubit_counts = vec![2, 4, 6, 8, 10, 12];
        let max_qubits = std::cmp::min(simulator.max_qubits(), 12);
        
        let mut results = Vec::new();
        
        for &num_qubits in &qubit_counts {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            let qft = GpuQuantumFourierTransform::new(num_qubits, false, backend);
            let start = Instant::now();
            qft.execute(&mut sim).await?;
            let duration = start.elapsed();
            
            let gate_count = num_qubits * (num_qubits + 1) / 2; // Approximate gate count for QFT
            results.push((num_qubits, gate_count, duration.as_millis()));
        }
        
        // Analyze QFT scaling
        tracing::info!("QFT scaling for {:?}:", backend);
        tracing::info!("Qubits | Gates | Time (ms) | ms/gate");
        for (qubits, gates, time_ms) in results {
            let ms_per_gate = time_ms as f64 / gates as f64;
            tracing::info!(
                "{:6} | {:5} | {:9} | {:8.3}",
                qubits, gates, time_ms, ms_per_gate
            );
        }
        
        Ok(())
    }
    
    // Helper functions for creating gates based on backend
    
    fn create_hadamard_gate(
        qubit: usize, 
        backend: GpuBackend
    ) -> Result<Box<dyn super::quantum_gpu::GpuQuantumGate>> {
        match backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Box::new(super::quantum_cuda_kernels::CudaQuantumGate::hadamard(qubit)))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(crate::QbmiaError::BackendNotSupported)
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Box::new(super::quantum_opencl_kernels::OpenClQuantumGate::hadamard(qubit)))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(crate::QbmiaError::BackendNotSupported)
                }
            }
            _ => Err(crate::QbmiaError::BackendNotSupported),
        }
    }
    
    fn create_cnot_gate(
        control: usize, 
        target: usize,
        backend: GpuBackend
    ) -> Result<Box<dyn super::quantum_gpu::GpuQuantumGate>> {
        match backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Box::new(super::quantum_cuda_kernels::CudaQuantumGate::cnot(control, target)))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(crate::QbmiaError::BackendNotSupported)
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Box::new(super::quantum_opencl_kernels::OpenClQuantumGate::cnot(control, target)))
                }
                #[cfg(not(feature = "opencl"))]
                {
                    Err(crate::QbmiaError::BackendNotSupported)
                }
            }
            _ => Err(crate::QbmiaError::BackendNotSupported),
        }
    }
}

/// Validation tests for GPU quantum implementations
pub struct GpuQuantumValidation;

impl GpuQuantumValidation {
    /// Run all validation tests
    pub async fn run_all_validation_tests() -> Result<()> {
        let simulators = GpuQuantumSimulatorFactory::detect_and_create_simulators().await?;
        
        for simulator in simulators {
            let backend = simulator.get_device_info().backend;
            tracing::info!("Running validation tests on {:?} backend", backend);
            
            Self::validate_single_qubit_gates(&simulator).await?;
            Self::validate_two_qubit_gates(&simulator).await?;
            Self::validate_qft_correctness(&simulator).await?;
            Self::validate_unitarity(&simulator).await?;
            Self::validate_measurement_statistics(&simulator).await?;
        }
        
        Ok(())
    }
    
    /// Validate single-qubit gate correctness
    async fn validate_single_qubit_gates(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        
        // Test Pauli gates
        let mut sim = simulator.clone();
        sim.initialize_qubits(1).await?;
        
        // Apply X gate twice should return to original state
        let x_gate = match backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Box::new(super::quantum_cuda_kernels::CudaQuantumGate::pauli_x(0)) 
                        as Box<dyn super::quantum_gpu::GpuQuantumGate>
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(crate::QbmiaError::BackendNotSupported);
                }
            }
            GpuBackend::OpenCL => {
                #[cfg(feature = "opencl")]
                {
                    Box::new(super::quantum_opencl_kernels::OpenClQuantumGate::pauli_x(0))
                        as Box<dyn super::quantum_gpu::GpuQuantumGate>
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Err(crate::QbmiaError::BackendNotSupported);
                }
            }
            _ => return Err(crate::QbmiaError::BackendNotSupported),
        };
        
        sim.apply_gate(&x_gate).await?;
        sim.apply_gate(&x_gate).await?;
        
        let state = sim.get_state_vector().unwrap();
        let fidelity = state[0].norm_sqr(); // Should be back to |0>
        assert!(
            fidelity > 0.99, 
            "Pauli-X gate validation failed on {:?}: fidelity = {}",
            backend, fidelity
        );
        
        tracing::info!("✓ Pauli gates validated on {:?} backend", backend);
        Ok(())
    }
    
    /// Validate two-qubit gate correctness
    async fn validate_two_qubit_gates(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        
        // Test Bell state creation
        let mut sim = simulator.clone();
        sim.initialize_qubits(2).await?;
        
        // Create Bell state: |00> + |11>
        let h_gate = GpuQuantumBenchmarks::create_hadamard_gate(0, backend)?;
        let cnot_gate = GpuQuantumBenchmarks::create_cnot_gate(0, 1, backend)?;
        
        sim.apply_gate(&h_gate).await?;
        sim.apply_gate(&cnot_gate).await?;
        
        let state = sim.get_state_vector().unwrap();
        let expected_amp = 1.0 / (2.0_f64).sqrt();
        
        // Check Bell state amplitudes
        assert!(
            (state[0].re - expected_amp).abs() < 1e-6,
            "Bell state |00> amplitude incorrect on {:?}: expected {}, got {}",
            backend, expected_amp, state[0].re
        );
        assert!(
            (state[3].re - expected_amp).abs() < 1e-6,
            "Bell state |11> amplitude incorrect on {:?}: expected {}, got {}",
            backend, expected_amp, state[3].re
        );
        assert!(
            state[1].norm() < 1e-6 && state[2].norm() < 1e-6,
            "Bell state should have zero amplitude for |01> and |10> on {:?}",
            backend
        );
        
        tracing::info!("✓ Two-qubit gates validated on {:?} backend", backend);
        Ok(())
    }
    
    /// Validate QFT mathematical properties
    async fn validate_qft_correctness(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        let max_qubits = std::cmp::min(simulator.max_qubits(), 8);
        
        for num_qubits in [2, 3, 4].iter().cloned() {
            if num_qubits > max_qubits { break; }
            
            let mut sim = simulator.clone();
            sim.initialize_qubits(num_qubits).await?;
            
            // Store initial state
            let initial_state = sim.get_state_vector().unwrap().clone();
            
            // Apply QFT
            let qft = GpuQuantumFourierTransform::new(num_qubits, false, backend);
            qft.execute(&mut sim).await?;
            
            // Apply inverse QFT
            let inv_qft = GpuQuantumFourierTransform::new(num_qubits, true, backend);
            inv_qft.execute(&mut sim).await?;
            
            // Should recover initial state
            let final_state = sim.get_state_vector().unwrap();
            let fidelity: f64 = initial_state.iter()
                .zip(final_state.iter())
                .map(|(a, b)| (a.conj() * b).re)
                .sum();
            
            assert!(
                fidelity > 0.99,
                "QFT unitarity test failed on {:?} with {} qubits: fidelity = {}",
                backend, num_qubits, fidelity
            );
        }
        
        tracing::info!("✓ QFT correctness validated on {:?} backend", backend);
        Ok(())
    }
    
    /// Validate gate unitarity
    async fn validate_unitarity(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        
        // Test that all gates preserve norm
        let mut sim = simulator.clone();
        sim.initialize_qubits(3).await?;
        
        // Prepare random superposition state
        let h_gates = vec![
            GpuQuantumBenchmarks::create_hadamard_gate(0, backend)?,
            GpuQuantumBenchmarks::create_hadamard_gate(1, backend)?,
            GpuQuantumBenchmarks::create_hadamard_gate(2, backend)?,
        ];
        
        for gate in h_gates {
            sim.apply_gate(&gate).await?;
        }
        
        let initial_norm: f64 = sim.get_state_vector().unwrap()
            .iter()
            .map(|c| c.norm_sqr())
            .sum();
        
        // Apply various gates
        let test_gates = vec![
            GpuQuantumBenchmarks::create_cnot_gate(0, 1, backend)?,
            GpuQuantumBenchmarks::create_cnot_gate(1, 2, backend)?,
            GpuQuantumBenchmarks::create_hadamard_gate(2, backend)?,
        ];
        
        for gate in test_gates {
            sim.apply_gate(&gate).await?;
            
            let current_norm: f64 = sim.get_state_vector().unwrap()
                .iter()
                .map(|c| c.norm_sqr())
                .sum();
            
            assert!(
                (current_norm - initial_norm).abs() < 1e-10,
                "Norm not preserved on {:?}: initial = {}, current = {}",
                backend, initial_norm, current_norm
            );
        }
        
        tracing::info!("✓ Gate unitarity validated on {:?} backend", backend);
        Ok(())
    }
    
    /// Validate measurement statistics
    async fn validate_measurement_statistics(simulator: &GpuQuantumSimulator) -> Result<()> {
        let backend = simulator.get_device_info().backend;
        
        // Test measurement of |+> state (50/50 probability)
        let num_trials = 1000;
        let mut zero_count = 0;
        
        for _ in 0..num_trials {
            let mut sim = simulator.clone();
            sim.initialize_qubits(1).await?;
            
            // Prepare |+> state
            let h_gate = GpuQuantumBenchmarks::create_hadamard_gate(0, backend)?;
            sim.apply_gate(&h_gate).await?;
            
            let measurements = sim.measure_all().await?;
            if !measurements[0] {
                zero_count += 1;
            }
        }
        
        let zero_probability = zero_count as f64 / num_trials as f64;
        let expected_probability = 0.5;
        let error = (zero_probability - expected_probability).abs();
        
        assert!(
            error < 0.05, // Allow 5% statistical error
            "Measurement statistics incorrect on {:?}: expected {}, got {} (error: {})",
            backend, expected_probability, zero_probability, error
        );
        
        tracing::info!(
            "✓ Measurement statistics validated on {:?} backend (p_0 = {:.3})",
            backend, zero_probability
        );
        Ok(())
    }
}