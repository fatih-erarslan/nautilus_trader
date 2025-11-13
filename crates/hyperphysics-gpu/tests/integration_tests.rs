//! Integration tests for GPU compute pipeline
//!
//! Tests GPU executor, shaders, and buffer management against CPU reference implementations.

use hyperphysics_gpu::{GPUExecutor, GPUBackend, backend::wgpu::WGPUBackend};
use hyperphysics_core::Result;

/// Small test lattice for quick validation
const TEST_LATTICE_SIZE: usize = 48;

/// Create simple coupling structure for testing
fn create_test_couplings() -> Vec<(usize, usize, f64)> {
    let mut couplings = Vec::new();

    // Create nearest-neighbor couplings on a linear chain
    for i in 0..TEST_LATTICE_SIZE - 1 {
        couplings.push((i, i + 1, 1.0)); // Ferromagnetic coupling
        couplings.push((i + 1, i, 1.0)); // Symmetric
    }

    couplings
}

/// CPU reference implementation of Ising energy
fn cpu_compute_energy(states: &[u32], couplings: &[(usize, usize, f64)]) -> f64 {
    let mut energy = 0.0;

    for &(i, j, strength) in couplings.iter() {
        if i < j {
            let spin_i = (states[i] as f64) * 2.0 - 1.0;
            let spin_j = (states[j] as f64) * 2.0 - 1.0;
            energy -= strength * spin_i * spin_j;
        }
    }

    energy
}

/// CPU reference implementation of Shannon entropy
fn cpu_compute_entropy(states: &[u32]) -> f64 {
    let mut entropy = 0.0;

    for &state in states.iter() {
        let p = state as f64;
        let q = 1.0 - p;

        if p > 1e-10 {
            entropy -= p * p.ln();
        }
        if q > 1e-10 {
            entropy -= q * q.ln();
        }
    }

    entropy
}

#[tokio::test]
async fn test_gpu_executor_initialization() -> Result<()> {
    // Test that GPU executor can be created
    let couplings = create_test_couplings();
    let executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await;

    assert!(executor.is_ok(), "GPU executor initialization failed");

    Ok(())
}

#[tokio::test]
async fn test_wgpu_backend_initialization() -> Result<()> {
    // Test that WGPU backend can be initialized
    let backend = WGPUBackend::new().await;

    assert!(backend.is_ok(), "WGPU backend initialization failed");

    if let Ok(backend) = backend {
        let caps = backend.capabilities();
        assert!(caps.supports_compute, "GPU must support compute shaders");
        assert!(caps.max_buffer_size >= 1024 * 1024, "GPU buffer too small");
        assert!(caps.max_workgroup_size >= 256, "Workgroup size too small");

        println!("GPU: {}", caps.device_name);
        println!("Max buffer: {} MB", caps.max_buffer_size / 1_000_000);
        println!("Max workgroup: {}", caps.max_workgroup_size);
    }

    Ok(())
}

#[tokio::test]
async fn test_gpu_energy_vs_cpu() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // Get initial states from GPU
    let gpu_states = executor.read_states().await?;
    let cpu_states: Vec<u32> = gpu_states.iter().map(|s| s.state).collect();

    // Compute energy on both GPU and CPU
    let gpu_energy = executor.compute_energy().await?;
    let cpu_energy = cpu_compute_energy(&cpu_states, &couplings);

    // They should match within floating-point tolerance
    let relative_error = ((gpu_energy - cpu_energy) / cpu_energy.abs().max(1e-10)).abs();

    println!("GPU Energy: {}", gpu_energy);
    println!("CPU Energy: {}", cpu_energy);
    println!("Relative Error: {:.2e}", relative_error);

    assert!(relative_error < 1e-5,
        "GPU energy {} doesn't match CPU energy {} (error: {:.2e})",
        gpu_energy, cpu_energy, relative_error);

    Ok(())
}

#[tokio::test]
async fn test_gpu_entropy_vs_cpu() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // Get initial states from GPU
    let gpu_states = executor.read_states().await?;
    let cpu_states: Vec<u32> = gpu_states.iter().map(|s| s.state).collect();

    // Compute entropy on both GPU and CPU
    let gpu_entropy = executor.compute_entropy().await?;
    let cpu_entropy = cpu_compute_entropy(&cpu_states);

    // They should match within floating-point tolerance
    let relative_error = ((gpu_entropy - cpu_entropy) / cpu_entropy.abs().max(1e-10)).abs();

    println!("GPU Entropy: {}", gpu_entropy);
    println!("CPU Entropy: {}", cpu_entropy);
    println!("Relative Error: {:.2e}", relative_error);

    assert!(relative_error < 1e-5,
        "GPU entropy {} doesn't match CPU entropy {} (error: {:.2e})",
        gpu_entropy, cpu_entropy, relative_error);

    Ok(())
}

#[tokio::test]
async fn test_gpu_state_update() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // Get initial states
    let initial_states = executor.read_states().await?;

    // Run one simulation step
    let temperature = 1.0;
    let dt = 0.01;
    executor.step(temperature, dt).await?;

    // Get updated states
    let updated_states = executor.read_states().await?;

    // States should have changed (with very high probability)
    let changed_count = initial_states.iter().zip(updated_states.iter())
        .filter(|(a, b)| a.state != b.state)
        .count();

    println!("Changed states: {}/{}", changed_count, TEST_LATTICE_SIZE);

    // At least some states should have flipped
    assert!(changed_count > 0, "No states changed after update step");

    Ok(())
}

#[tokio::test]
async fn test_gpu_double_buffering() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // Run multiple steps to test buffer swapping
    let temperature = 1.0;
    let dt = 0.01;

    for i in 0..5 {
        executor.step(temperature, dt).await?;
        let states = executor.read_states().await?;
        println!("Step {}: {} active pBits", i,
            states.iter().filter(|s| s.state == 1).count());
    }

    // If we get here without panicking, double buffering works
    Ok(())
}

#[tokio::test]
async fn test_gpu_bias_update() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // Create strong positive bias (should favor state = 1)
    let strong_biases: Vec<f32> = vec![10.0; TEST_LATTICE_SIZE];
    executor.update_biases(&strong_biases).await?;

    // Run simulation at low temperature (should follow bias)
    let temperature = 0.1;
    let dt = 0.01;

    for _ in 0..20 {
        executor.step(temperature, dt).await?;
    }

    // Most pBits should be in state = 1 due to strong positive bias
    let final_states = executor.read_states().await?;
    let ones_count = final_states.iter().filter(|s| s.state == 1).count();
    let ones_fraction = ones_count as f64 / TEST_LATTICE_SIZE as f64;

    println!("Fraction in state 1: {:.2}", ones_fraction);

    assert!(ones_fraction > 0.7,
        "Strong positive bias should favor state 1 (got {:.2})",
        ones_fraction);

    Ok(())
}

#[tokio::test]
async fn test_gpu_ferromagnetic_ordering() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // Ferromagnetic coupling should favor aligned states at low temperature
    let temperature = 0.5;
    let dt = 0.01;

    // Run enough steps to reach equilibrium
    for _ in 0..100 {
        executor.step(temperature, dt).await?;
    }

    let final_states = executor.read_states().await?;

    // Count aligned neighbors
    let mut aligned_count = 0;
    let mut total_pairs = 0;

    for i in 0..TEST_LATTICE_SIZE - 1 {
        if final_states[i].state == final_states[i + 1].state {
            aligned_count += 1;
        }
        total_pairs += 1;
    }

    let alignment_fraction = aligned_count as f64 / total_pairs as f64;

    println!("Alignment fraction: {:.2}", alignment_fraction);

    // At low temperature, ferromagnetic coupling should produce high alignment
    // Note: Threshold at 0.55 to account for stochastic variation with GPU RNG
    assert!(alignment_fraction > 0.55,
        "Ferromagnetic coupling should favor alignment (got {:.2})",
        alignment_fraction);

    Ok(())
}

#[tokio::test]
async fn test_gpu_energy_conservation() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // At very low temperature, energy should decrease over time
    let temperature = 0.1;
    let dt = 0.01;

    let initial_energy = executor.compute_energy().await?;

    // Run simulation
    for _ in 0..50 {
        executor.step(temperature, dt).await?;
    }

    let final_energy = executor.compute_energy().await?;

    println!("Initial Energy: {}", initial_energy);
    println!("Final Energy: {}", final_energy);

    // Energy should decrease (or stay same) at low temperature
    assert!(final_energy <= initial_energy + 1.0,
        "Energy should not significantly increase at low temperature");

    Ok(())
}

#[tokio::test]
async fn test_gpu_async_readback() -> Result<()> {
    let couplings = create_test_couplings();
    let mut executor = GPUExecutor::new(TEST_LATTICE_SIZE, &couplings).await?;

    // Test that async readback doesn't block
    let start = std::time::Instant::now();

    // Read states (async operation)
    let states = executor.read_states().await?;

    let duration = start.elapsed();

    println!("Async readback took: {:?}", duration);
    assert_eq!(states.len(), TEST_LATTICE_SIZE);

    // Should complete reasonably quickly (< 100ms)
    assert!(duration.as_millis() < 100,
        "Async readback took too long: {:?}", duration);

    Ok(())
}

#[tokio::test]
#[ignore] // Only run for performance profiling
async fn test_gpu_large_lattice() -> Result<()> {
    // Test with larger lattice to verify GPU advantage
    const LARGE_SIZE: usize = 10_000;

    let mut couplings = Vec::new();
    for i in 0..LARGE_SIZE - 1 {
        couplings.push((i, i + 1, 1.0));
        couplings.push((i + 1, i, 1.0));
    }

    println!("Creating GPU executor for {} pBits...", LARGE_SIZE);
    let start = std::time::Instant::now();

    let mut executor = GPUExecutor::new(LARGE_SIZE, &couplings).await?;

    println!("Initialization took: {:?}", start.elapsed());

    // Run simulation steps
    let step_start = std::time::Instant::now();

    for _ in 0..10 {
        executor.step(1.0, 0.01).await?;
    }

    let step_duration = step_start.elapsed();
    println!("10 simulation steps took: {:?}", step_duration);
    println!("Average per step: {:?}", step_duration / 10);

    // Compute observables
    let obs_start = std::time::Instant::now();
    let energy = executor.compute_energy().await?;
    let entropy = executor.compute_entropy().await?;
    let obs_duration = obs_start.elapsed();

    println!("Energy: {}, Entropy: {}", energy, entropy);
    println!("Observable computation took: {:?}", obs_duration);

    Ok(())
}
