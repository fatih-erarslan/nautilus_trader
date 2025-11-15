//! CPU-based compute executor (fallback when GPU unavailable)
//!
//! Provides the same functionality as GPUExecutor but uses CPU parallelism
//! via Rayon instead of GPU compute shaders.

use super::executor::{GPUPBitState, GPUCoupling};
use hyperphysics_core::Result;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// CPU-based compute executor using Rayon for parallelism
pub struct CPUExecutor {
    lattice_size: usize,

    // Double buffering for pBit states
    state_buffer_a: Vec<GPUPBitState>,
    state_buffer_b: Vec<GPUPBitState>,
    current_buffer: usize,

    // Coupling topology
    couplings: Vec<GPUCoupling>,

    // Simulation parameters
    temperature: f32,
    dt: f32,

    // RNG state (simple linear congruential)
    rng_state: Arc<Mutex<u64>>,
}

impl CPUExecutor {
    /// Create new CPU executor
    ///
    /// # Arguments
    /// * `lattice_size` - Number of pBits in the lattice
    /// * `couplings` - Coupling topology as (source, target, strength) triples
    pub fn new(
        lattice_size: usize,
        couplings: &[(usize, usize, f64)],
    ) -> Result<Self> {
        // Build coupling buffer and initial states
        let (gpu_couplings, initial_states) = Self::build_coupling_buffer(lattice_size, couplings);

        Ok(Self {
            lattice_size,
            state_buffer_a: initial_states.clone(),
            state_buffer_b: initial_states,
            current_buffer: 0,
            couplings: gpu_couplings,
            temperature: 1.0,
            dt: 0.01,
            rng_state: Arc::new(Mutex::new(12345)), // Default seed
        })
    }

    /// Build coupling buffer from edge list
    fn build_coupling_buffer(
        lattice_size: usize,
        couplings: &[(usize, usize, f64)],
    ) -> (Vec<GPUCoupling>, Vec<GPUPBitState>) {
        let mut gpu_couplings = Vec::new();
        let mut states = vec![
            GPUPBitState {
                state: 0,
                bias: 0.0,
                coupling_offset: 0,
                coupling_count: 0,
            };
            lattice_size
        ];

        // Group couplings by source node
        let mut coupling_groups: Vec<Vec<(usize, f64)>> = vec![Vec::new(); lattice_size];
        for &(source, target, strength) in couplings {
            if source < lattice_size && target < lattice_size {
                coupling_groups[source].push((target, strength));
            }
        }

        // Build flat coupling buffer with offsets
        let mut offset = 0;
        for (source, targets) in coupling_groups.iter().enumerate() {
            states[source].coupling_offset = offset as u32;
            states[source].coupling_count = targets.len() as u32;

            for &(target, strength) in targets {
                gpu_couplings.push(GPUCoupling {
                    target_idx: target as u32,
                    strength: strength as f32,
                });
            }

            offset += targets.len();
        }

        (gpu_couplings, states)
    }

    /// Update pBit states (CPU parallel implementation)
    pub fn update_states(&mut self, temperature: f32, dt: f32) -> Result<()> {
        self.temperature = temperature;
        self.dt = dt;

        let (read_buffer, write_buffer) = if self.current_buffer == 0 {
            (&self.state_buffer_a, &mut self.state_buffer_b)
        } else {
            (&self.state_buffer_b, &mut self.state_buffer_a)
        };

        let couplings = &self.couplings;
        let rng_state = Arc::clone(&self.rng_state);

        // Parallel update using Rayon
        write_buffer.par_iter_mut().enumerate().for_each(|(i, new_state)| {
            let old_state = &read_buffer[i];

            // Calculate local field from couplings
            let offset = old_state.coupling_offset as usize;
            let count = old_state.coupling_count as usize;

            let mut local_field = old_state.bias;
            for j in 0..count {
                let coupling = &couplings[offset + j];
                let neighbor_state = read_buffer[coupling.target_idx as usize].state;
                let spin = if neighbor_state == 1 { 1.0 } else { -1.0 };
                local_field += coupling.strength * spin;
            }

            // Compute transition probability (sigmoid)
            let prob = 1.0 / (1.0 + (-local_field / temperature).exp());

            // Sample new state (thread-safe RNG)
            let rand_val = {
                let mut state = rng_state.lock().unwrap();
                *state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (*state / 65536) % 32768
            };
            let rand_float = (rand_val as f32) / 32768.0;

            new_state.state = if rand_float < prob { 1 } else { 0 };
            new_state.bias = old_state.bias;
            new_state.coupling_offset = old_state.coupling_offset;
            new_state.coupling_count = old_state.coupling_count;
        });

        // Swap buffers
        self.current_buffer = 1 - self.current_buffer;

        Ok(())
    }

    /// Compute total energy (CPU parallel implementation)
    pub fn compute_energy(&self) -> Result<f64> {
        let states = if self.current_buffer == 0 {
            &self.state_buffer_a
        } else {
            &self.state_buffer_b
        };

        let couplings = &self.couplings;

        // Parallel energy computation using Rayon
        let energy: f64 = states
            .par_iter()
            .enumerate()
            .map(|(i, state)| {
                let offset = state.coupling_offset as usize;
                let count = state.coupling_count as usize;

                let spin_i = if state.state == 1 { 1.0_f64 } else { -1.0_f64 };
                let mut local_energy = 0.0_f64;

                for j in 0..count {
                    let coupling = &couplings[offset + j];
                    let neighbor_state = states[coupling.target_idx as usize].state;
                    let spin_j = if neighbor_state == 1 { 1.0_f64 } else { -1.0_f64 };

                    // Avoid double counting by only counting i < j
                    if (coupling.target_idx as usize) > i {
                        local_energy -= (coupling.strength as f64) * spin_i * spin_j;
                    }
                }

                local_energy
            })
            .sum();

        Ok(energy)
    }

    /// Compute Shannon entropy (CPU parallel implementation)
    pub fn compute_entropy(&self) -> Result<f64> {
        let states = if self.current_buffer == 0 {
            &self.state_buffer_a
        } else {
            &self.state_buffer_b
        };

        // Parallel entropy computation using Rayon
        let entropy: f64 = states
            .par_iter()
            .map(|state| {
                let p = state.state as f64;
                let q = 1.0 - p;

                let mut local_entropy = 0.0;
                if p > 1e-10 {
                    local_entropy -= p * p.ln();
                }
                if q > 1e-10 {
                    local_entropy -= q * q.ln();
                }

                local_entropy
            })
            .sum();

        Ok(entropy)
    }

    /// Read current pBit states
    pub fn read_states(&self) -> Result<Vec<GPUPBitState>> {
        let states = if self.current_buffer == 0 {
            &self.state_buffer_a
        } else {
            &self.state_buffer_b
        };

        Ok(states.clone())
    }

    /// Update bias for a specific pBit
    pub fn update_bias(&mut self, pbit_idx: usize, new_bias: f32) -> Result<()> {
        if pbit_idx >= self.lattice_size {
            return Err(hyperphysics_core::EngineError::Simulation {
                message: format!("Invalid pBit index: {} (max: {})", pbit_idx, self.lattice_size - 1),
            });
        }

        // Update both buffers to maintain consistency
        self.state_buffer_a[pbit_idx].bias = new_bias;
        self.state_buffer_b[pbit_idx].bias = new_bias;

        Ok(())
    }

    /// Get lattice size
    pub fn lattice_size(&self) -> usize {
        self.lattice_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_couplings(size: usize) -> Vec<(usize, usize, f64)> {
        let mut couplings = Vec::new();
        for i in 0..size - 1 {
            couplings.push((i, i + 1, 1.0)); // Forward
            couplings.push((i + 1, i, 1.0)); // Backward
        }
        couplings
    }

    #[test]
    fn test_cpu_executor_creation() {
        let couplings = create_test_couplings(10);
        let executor = CPUExecutor::new(10, &couplings);

        assert!(executor.is_ok());
        let executor = executor.unwrap();
        assert_eq!(executor.lattice_size(), 10);
    }

    #[test]
    fn test_cpu_energy_computation() {
        let couplings = create_test_couplings(10);
        let executor = CPUExecutor::new(10, &couplings).unwrap();

        let energy = executor.compute_energy();
        assert!(energy.is_ok());

        // Energy should be finite
        let energy_val = energy.unwrap();
        assert!(energy_val.is_finite());
    }

    #[test]
    fn test_cpu_entropy_computation() {
        let couplings = create_test_couplings(10);
        let executor = CPUExecutor::new(10, &couplings).unwrap();

        let entropy = executor.compute_entropy();
        assert!(entropy.is_ok());

        // Entropy should be non-negative
        let entropy_val = entropy.unwrap();
        assert!(entropy_val >= 0.0);
    }

    #[test]
    fn test_cpu_state_update() {
        let couplings = create_test_couplings(10);
        let mut executor = CPUExecutor::new(10, &couplings).unwrap();

        let result = executor.update_states(1.0, 0.01);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cpu_bias_update() {
        let couplings = create_test_couplings(10);
        let mut executor = CPUExecutor::new(10, &couplings).unwrap();

        let result = executor.update_bias(5, 0.5);
        assert!(result.is_ok());

        // Invalid index should fail
        let result = executor.update_bias(100, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_read_states() {
        let couplings = create_test_couplings(10);
        let executor = CPUExecutor::new(10, &couplings).unwrap();

        let states = executor.read_states();
        assert!(states.is_ok());

        let states = states.unwrap();
        assert_eq!(states.len(), 10);
    }
}
