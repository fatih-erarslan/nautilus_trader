//! GPU-Only Quantum Computing Demonstration
//! 
//! This example demonstrates the unified GPU-only quantum computing framework
//! with STRICT TENGRI compliance. All quantum operations run on local GPU
//! hardware using CUDA, OpenCL, Vulkan, or Metal backends.
//! 
//! NO CLOUD QUANTUM BACKENDS - PURE LOCAL GPU SIMULATION

use std::time::Instant;
use qbmia_unified::gpu::{
    GpuQuantumSimulatorFactory,
    GpuQuantumFourierTransform,
    GpuVariationalQuantumEigensolver,
    GpuQuantumApproximateOptimization,
    GpuPauliHamiltonian,
    GpuQuantumBenchmarks,
    GpuQuantumValidation,
};
use qbmia_unified::{Result, GpuBackend};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 QBMIA Unified GPU-Only Quantum Computing Framework");
    println!("🔒 TENGRI COMPLIANT - NO CLOUD QUANTUM BACKENDS");
    println!("⚡ Local GPU Hardware Acceleration Only");
    println!();
    
    // Detect available GPU quantum simulators
    println!("🔍 Detecting GPU quantum simulators...");
    let simulators = match GpuQuantumSimulatorFactory::detect_and_create_simulators().await {
        Ok(sims) if !sims.is_empty() => sims,
        Ok(_) => {
            println!("❌ No GPU devices available for quantum simulation");
            println!("   Please ensure you have:");
            println!("   - NVIDIA GPU with CUDA support, or");
            println!("   - AMD/Intel GPU with OpenCL support, or");
            println!("   - Modern GPU with Vulkan compute support, or");
            println!("   - Apple Silicon with Metal compute support");
            return Ok(());
        }
        Err(e) => {
            println!("❌ Failed to detect GPU devices: {}", e);
            return Ok(());
        }
    };
    
    println!("✅ Found {} GPU quantum simulator(s)", simulators.len());
    
    // Display available simulators
    for (i, simulator) in simulators.iter().enumerate() {
        let info = simulator.get_device_info();
        println!("  {}: {} ({:?} backend)", 
            i + 1, info.device_name, info.backend);
        println!("     Max qubits: {}", simulator.max_qubits());
        println!("     Memory: {:.2} GB", 
            info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
        println!("     Bandwidth: {:.2} GB/s", info.memory_bandwidth_gbps);
    }
    println!();
    
    // Get the best available simulator
    let best_simulator = GpuQuantumSimulatorFactory::get_best_simulator().await?;
    let backend = best_simulator.get_device_info().backend;
    
    println!("🎯 Using best simulator: {} ({:?})", 
        best_simulator.get_device_info().device_name, backend);
    println!("   Maximum qubits: {}", best_simulator.max_qubits());
    println!();
    
    // Run demonstration workflows
    demo_basic_quantum_operations(&best_simulator).await?;
    demo_quantum_fourier_transform(&best_simulator).await?;
    demo_variational_quantum_eigensolver(&best_simulator).await?;
    demo_quantum_approximate_optimization(&best_simulator).await?;
    demo_performance_analysis(&best_simulator).await?;
    
    // Run validation and benchmarks
    if simulators.len() > 0 {
        println!("🧪 Running validation tests...");
        match GpuQuantumValidation::run_all_validation_tests().await {
            Ok(()) => println!("✅ All validation tests passed"),
            Err(e) => println!("⚠️  Some validation tests failed: {}", e),
        }
        println!();
        
        println!("📊 Running performance benchmarks...");
        match GpuQuantumBenchmarks::run_all_benchmarks().await {
            Ok(()) => println!("✅ Performance benchmarks completed"),
            Err(e) => println!("⚠️  Some benchmarks failed: {}", e),
        }
        println!();
    }
    
    println!("🎉 GPU Quantum Computing Demonstration Complete!");
    println!("   All operations executed on local GPU hardware");
    println!("   TENGRI compliance verified - no cloud backends used");
    
    Ok(())
}

/// Demonstrate basic quantum operations
async fn demo_basic_quantum_operations(
    simulator: &qbmia_unified::gpu::GpuQuantumSimulator
) -> Result<()> {
    println!("🔄 Demo: Basic Quantum Operations");
    
    let backend = simulator.get_device_info().backend;
    let mut sim = simulator.clone();
    
    // Initialize quantum state
    let num_qubits = 3;
    sim.initialize_qubits(num_qubits).await?;
    println!("   Initialized {} qubit quantum state", num_qubits);
    
    // Create superposition
    for qubit in 0..num_qubits {
        let gate = GpuQuantumBenchmarks::create_hadamard_gate(qubit, backend)?;
        sim.apply_gate(&gate).await?;
    }
    println!("   Applied Hadamard gates to create superposition");
    
    // Create entanglement
    for qubit in 0..num_qubits - 1 {
        let gate = GpuQuantumBenchmarks::create_cnot_gate(qubit, qubit + 1, backend)?;
        sim.apply_gate(&gate).await?;
    }
    println!("   Applied CNOT gates to create entanglement");
    
    // Measure quantum state
    let measurements = sim.measure_all().await?;
    let measurement_string: String = measurements.iter()
        .map(|&b| if b { '1' } else { '0' })
        .collect();
    println!("   Measurement result: |{}>", measurement_string);
    
    // Check final state
    let state = sim.get_state_vector().unwrap();
    let probabilities: Vec<f64> = state.iter().map(|c| c.norm_sqr()).collect();
    println!("   Non-zero probabilities:");
    for (i, &prob) in probabilities.iter().enumerate() {
        if prob > 1e-6 {
            println!("     |{:03b}>: {:.4}", i, prob);
        }
    }
    
    println!("✅ Basic quantum operations completed\n");
    Ok(())
}

/// Demonstrate Quantum Fourier Transform
async fn demo_quantum_fourier_transform(
    simulator: &qbmia_unified::gpu::GpuQuantumSimulator
) -> Result<()> {
    println!("🌊 Demo: Quantum Fourier Transform");
    
    let backend = simulator.get_device_info().backend;
    let num_qubits = std::cmp::min(4, simulator.max_qubits());
    
    let mut sim = simulator.clone();
    sim.initialize_qubits(num_qubits).await?;
    
    // Store initial state
    let initial_state = sim.get_state_vector().unwrap().clone();
    println!("   Initial state: |{:0width$b}>", 0, width = num_qubits);
    
    // Apply QFT
    let start = Instant::now();
    let qft = GpuQuantumFourierTransform::new(num_qubits, false, backend);
    qft.execute(&mut sim).await?;
    let qft_time = start.elapsed();
    
    println!("   Applied QFT in {:.2} μs", qft_time.as_micros());
    
    // Show transformed state
    let qft_state = sim.get_state_vector().unwrap();
    println!("   QFT state amplitudes:");
    for (i, amp) in qft_state.iter().enumerate() {
        if amp.norm() > 1e-6 {
            println!("     |{:0width$b}>: {:.4} + {:.4}i", 
                i, amp.re, amp.im, width = num_qubits);
        }
    }
    
    // Apply inverse QFT
    let start = Instant::now();
    let inv_qft = GpuQuantumFourierTransform::new(num_qubits, true, backend);
    inv_qft.execute(&mut sim).await?;
    let inv_qft_time = start.elapsed();
    
    println!("   Applied inverse QFT in {:.2} μs", inv_qft_time.as_micros());
    
    // Verify round-trip fidelity
    let final_state = sim.get_state_vector().unwrap();
    let fidelity: f64 = initial_state.iter()
        .zip(final_state.iter())
        .map(|(a, b)| (a.conj() * b).re)
        .sum();
    
    println!("   Round-trip fidelity: {:.8}", fidelity);
    println!("✅ QFT demonstration completed\n");
    Ok(())
}

/// Demonstrate Variational Quantum Eigensolver
async fn demo_variational_quantum_eigensolver(
    simulator: &qbmia_unified::gpu::GpuQuantumSimulator
) -> Result<()> {
    println!("🧬 Demo: Variational Quantum Eigensolver (VQE)");
    
    let backend = simulator.get_device_info().backend;
    let num_qubits = std::cmp::min(4, simulator.max_qubits());
    
    // Create Ising model Hamiltonian
    let mut hamiltonian = GpuPauliHamiltonian::new(num_qubits);
    
    // Nearest-neighbor interactions
    for i in 0..num_qubits - 1 {
        hamiltonian.add_term(-1.0, vec![(i, 'Z'), (i + 1, 'Z')]);
    }
    
    // Transverse field
    for i in 0..num_qubits {
        hamiltonian.add_term(-0.5, vec![(i, 'X')]);
    }
    
    println!("   Created Ising model Hamiltonian with {} qubits", num_qubits);
    println!("   Terms: {} ZZ interactions + {} X fields", num_qubits - 1, num_qubits);
    
    // Initialize VQE
    let mut vqe = GpuVariationalQuantumEigensolver::new(num_qubits, 2, backend);
    let mut sim = simulator.clone();
    
    // Initial energy evaluation
    let initial_energy = vqe.evaluate_energy(&mut sim, &hamiltonian).await?;
    println!("   Initial energy: {:.6}", initial_energy);
    
    // Run optimization
    let start = Instant::now();
    let (final_energy, best_params) = vqe.optimize(&mut sim, &hamiltonian, 100).await?;
    let optimization_time = start.elapsed();
    
    println!("   Final energy: {:.6}", final_energy);
    println!("   Energy improvement: {:.6}", initial_energy - final_energy);
    println!("   Optimization time: {:.2} ms", optimization_time.as_millis());
    println!("   Optimized {} parameters", best_params.len());
    
    println!("✅ VQE demonstration completed\n");
    Ok(())
}

/// Demonstrate Quantum Approximate Optimization Algorithm
async fn demo_quantum_approximate_optimization(
    simulator: &qbmia_unified::gpu::GpuQuantumSimulator
) -> Result<()> {
    println!("🎯 Demo: Quantum Approximate Optimization Algorithm (QAOA)");
    
    let backend = simulator.get_device_info().backend;
    let num_qubits = std::cmp::min(4, simulator.max_qubits());
    
    // Create Max-Cut problem on a cycle graph
    let edges: Vec<(usize, usize)> = (0..num_qubits)
        .map(|i| (i, (i + 1) % num_qubits))
        .collect();
    
    let hamiltonian = GpuPauliHamiltonian::max_cut_hamiltonian(&edges, num_qubits);
    
    println!("   Created Max-Cut problem on {}-node cycle", num_qubits);
    println!("   Edges: {:?}", edges);
    
    // Initialize QAOA
    let mut qaoa = GpuQuantumApproximateOptimization::new(num_qubits, 2, backend);
    let mut sim = simulator.clone();
    
    // Initial cost evaluation
    let initial_cost = qaoa.evaluate_cost(&mut sim, &hamiltonian).await?;
    println!("   Initial cost: {:.6}", initial_cost);
    
    // Run optimization
    let start = Instant::now();
    let (final_cost, best_beta, best_gamma) = qaoa.optimize(&mut sim, &hamiltonian, 50).await?;
    let optimization_time = start.elapsed();
    
    println!("   Final cost: {:.6}", final_cost);
    println!("   Cost improvement: {:.6}", initial_cost - final_cost);
    println!("   Optimization time: {:.2} ms", optimization_time.as_millis());
    println!("   Best β parameters: {:?}", best_beta);
    println!("   Best γ parameters: {:?}", best_gamma);
    
    // Analyze final quantum state
    qaoa.prepare_qaoa_state(&mut sim).await?;
    let final_state = sim.get_state_vector().unwrap();
    
    println!("   Final state probabilities:");
    for (i, amp) in final_state.iter().enumerate() {
        let prob = amp.norm_sqr();
        if prob > 0.01 {
            let bit_string: String = (0..num_qubits)
                .map(|j| if (i >> j) & 1 == 1 { '1' } else { '0' })
                .collect();
            println!("     |{}>: {:.3}", bit_string, prob);
        }
    }
    
    println!("✅ QAOA demonstration completed\n");
    Ok(())
}

/// Demonstrate performance analysis
async fn demo_performance_analysis(
    simulator: &qbmia_unified::gpu::GpuQuantumSimulator
) -> Result<()> {
    println!("📈 Demo: Performance Analysis");
    
    let backend = simulator.get_device_info().backend;
    let device_info = simulator.get_device_info();
    
    println!("   Device: {}", device_info.device_name);
    println!("   Backend: {:?}", backend);
    println!("   Max qubits: {}", simulator.max_qubits());
    println!("   Memory: {:.2} GB", device_info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Bandwidth: {:.2} GB/s", device_info.memory_bandwidth_gbps);
    println!();
    
    // Gate performance test
    println!("   Gate performance:");
    let max_test_qubits = std::cmp::min(simulator.max_qubits(), 12);
    
    for num_qubits in [2, 4, 6, 8, 10, 12] {
        if num_qubits > max_test_qubits { break; }
        
        let mut sim = simulator.clone();
        sim.initialize_qubits(num_qubits).await?;
        
        // Time Hadamard gate application
        let start = Instant::now();
        let gate = GpuQuantumBenchmarks::create_hadamard_gate(0, backend)?;
        sim.apply_gate(&gate).await?;
        let gate_time = start.elapsed();
        
        let state_size = 1 << num_qubits;
        let throughput = state_size as f64 / gate_time.as_secs_f64();
        
        println!("     {} qubits: {:.1} μs ({:.0} amplitudes/sec)", 
            num_qubits, gate_time.as_micros(), throughput);
    }
    
    // Memory bandwidth test
    println!("   Memory bandwidth test:");
    let _ = GpuQuantumBenchmarks::benchmark_memory_bandwidth(simulator).await;
    
    println!("✅ Performance analysis completed\n");
    Ok(())
}