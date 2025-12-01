//! # Metropolis-Hastings Sweeper
//!
//! Ultra-fast Metropolis dynamics implementation for pBit arrays.
//! Optimized for cache efficiency and minimal branching.

use super::{PackedPBitArray, SparseCouplings, BITS_PER_WORD};
use std::time::Instant;

/// Result of a Metropolis sweep
#[derive(Debug, Clone)]
pub struct SweepResult {
    /// Number of accepted flips
    pub flips: u32,
    /// Duration in nanoseconds
    pub duration_ns: u64,
    /// Acceptance rate
    pub acceptance_rate: f32,
}

/// High-performance Metropolis sweeper
pub struct MetropolisSweeper {
    /// Current temperature
    temperature: f64,
    /// Inverse temperature (beta = 1/T)
    beta: f64,
    /// Fast random number generator
    rng: fastrand::Rng,
    /// Precomputed acceptance thresholds for common ΔE values
    acceptance_table: [f32; 256],
}

impl MetropolisSweeper {
    /// Create a new sweeper at given temperature
    pub fn new(temperature: f64) -> Self {
        let mut sweeper = Self {
            temperature,
            beta: 1.0 / temperature,
            rng: fastrand::Rng::new(),
            acceptance_table: [0.0; 256],
        };
        sweeper.rebuild_acceptance_table();
        sweeper
    }

    /// Set temperature and rebuild lookup table
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
        self.beta = 1.0 / temperature;
        self.rebuild_acceptance_table();
    }

    /// Precompute acceptance probabilities for common ΔE values
    fn rebuild_acceptance_table(&mut self) {
        // Table covers ΔE from -8.0 to +8.0 in steps of ~0.0625
        for i in 0..256 {
            let delta_e = (i as f64 - 128.0) * 0.0625;
            self.acceptance_table[i] = (-self.beta * delta_e).exp().min(1.0) as f32;
        }
    }

    /// Fast acceptance probability lookup
    #[inline(always)]
    fn acceptance_prob(&self, delta_e: f32) -> f32 {
        if delta_e <= 0.0 {
            return 1.0; // Always accept energy-lowering moves
        }
        
        // Use lookup table for common range
        let idx = ((delta_e * 16.0 + 128.0) as usize).min(255);
        self.acceptance_table[idx]
    }

    /// Fast uniform random in [0, 1)
    #[inline(always)]
    fn random_uniform(&mut self) -> f32 {
        self.rng.f32()
    }

    /// Perform a single Metropolis sweep (sequential updates)
    ///
    /// Uses sequential order for maximum cache efficiency.
    /// For truly random updates, use `sweep_random`.
    pub fn sweep(
        &mut self,
        states: &mut PackedPBitArray,
        biases: &[f32],
        couplings: &SparseCouplings,
        _cached_fields: &mut [f32],
    ) -> SweepResult {
        let start = Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Sequential sweep for cache efficiency
        for i in 0..n {
            // Calculate energy change for flipping pBit i
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            
            // Metropolis acceptance criterion
            if delta_e <= 0.0 || self.random_uniform() < self.acceptance_prob(delta_e) {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SweepResult {
            flips,
            duration_ns: duration.as_nanos() as u64,
            acceptance_rate: flips as f32 / n as f32,
        }
    }

    /// Perform a single Metropolis sweep with random order
    ///
    /// More accurate for physics but slower due to random access.
    pub fn sweep_random(
        &mut self,
        states: &mut PackedPBitArray,
        biases: &[f32],
        couplings: &SparseCouplings,
    ) -> SweepResult {
        let start = Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Random order sweep (slower but more accurate)
        for _ in 0..n {
            let i = self.rng.usize(0..n);
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            
            if delta_e <= 0.0 || self.random_uniform() < self.acceptance_prob(delta_e) {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SweepResult {
            flips,
            duration_ns: duration.as_nanos() as u64,
            acceptance_rate: flips as f32 / n as f32,
        }
    }

    /// Perform a checkerboard sweep (for parallelization)
    ///
    /// Updates all "even" pBits first, then all "odd" pBits.
    /// This allows parallel updates within each phase.
    pub fn checkerboard_sweep(
        &mut self,
        states: &mut PackedPBitArray,
        biases: &[f32],
        couplings: &SparseCouplings,
    ) -> SweepResult {
        let start = Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Phase 1: Update even indices
        for i in (0..n).step_by(2) {
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            if delta_e <= 0.0 || self.random_uniform() < self.acceptance_prob(delta_e) {
                states.flip(i);
                flips += 1;
            }
        }

        // Phase 2: Update odd indices
        for i in (1..n).step_by(2) {
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            if delta_e <= 0.0 || self.random_uniform() < self.acceptance_prob(delta_e) {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SweepResult {
            flips,
            duration_ns: duration.as_nanos() as u64,
            acceptance_rate: flips as f32 / n as f32,
        }
    }

    /// Perform batch update on a word (64 pBits at once)
    ///
    /// For very high-temperature or uncoupled systems.
    pub fn batch_word_update(
        &mut self,
        states: &mut PackedPBitArray,
        word_idx: usize,
    ) -> u32 {
        // At high T, each bit has ~50% chance of flipping
        let flip_mask = self.rng.u64(..);
        let old_word = states.get_word(word_idx);
        let new_word = old_word ^ flip_mask;
        states.set_word(word_idx, new_word);
        (old_word ^ new_word).count_ones()
    }

    /// Fast single-spin update (for tight loops)
    #[inline(always)]
    pub fn single_update(
        &mut self,
        states: &PackedPBitArray,
        idx: usize,
        delta_e: f32,
    ) -> bool {
        if delta_e <= 0.0 {
            true
        } else {
            self.random_uniform() < self.acceptance_prob(delta_e)
        }
    }

    /// Get current temperature
    #[inline]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get inverse temperature
    #[inline]
    pub fn beta(&self) -> f64 {
        self.beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::SparseCouplings;

    #[test]
    fn test_sweep_basic() {
        let n = 100;
        let mut states = PackedPBitArray::random(n, 42);
        let biases = vec![0.0f32; n];
        
        // Create simple nearest-neighbor couplings
        let mut couplings = SparseCouplings::new(n);
        for i in 0..n {
            let j = (i + 1) % n;
            couplings.add(i, j, 1.0);
            couplings.add(j, i, 1.0);
        }
        couplings.finalize();
        
        let mut cached = vec![0.0f32; n];
        let mut sweeper = MetropolisSweeper::new(1.0);
        
        let result = sweeper.sweep(&mut states, &biases, &couplings, &mut cached);
        
        assert!(result.flips > 0);
        assert!(result.acceptance_rate > 0.0 && result.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_temperature_effect() {
        let n = 1000;
        let mut states_low = PackedPBitArray::new(n);
        let mut states_high = PackedPBitArray::new(n);
        let biases = vec![0.0f32; n];
        
        let mut couplings = SparseCouplings::new(n);
        for i in 0..n {
            couplings.add(i, (i + 1) % n, 1.0);
        }
        couplings.finalize();
        
        let mut cached = vec![0.0f32; n];
        
        // Low temperature: fewer flips
        let mut sweeper_low = MetropolisSweeper::new(0.1);
        let result_low = sweeper_low.sweep(&mut states_low, &biases, &couplings, &mut cached);
        
        // High temperature: more flips
        let mut sweeper_high = MetropolisSweeper::new(10.0);
        let result_high = sweeper_high.sweep(&mut states_high, &biases, &couplings, &mut cached);
        
        // At high T, should have higher acceptance rate
        println!("Low T acceptance: {:.3}", result_low.acceptance_rate);
        println!("High T acceptance: {:.3}", result_high.acceptance_rate);
        
        assert!(result_high.acceptance_rate > result_low.acceptance_rate);
    }

    #[test]
    fn test_sweep_performance() {
        let n = 10_000;
        let mut states = PackedPBitArray::random(n, 42);
        let biases = vec![0.0f32; n];
        
        // Sparse random couplings
        let mut couplings = SparseCouplings::with_capacity(n, n * 10);
        let mut rng = fastrand::Rng::with_seed(123);
        for _ in 0..(n * 5) {
            let i = rng.usize(0..n);
            let j = rng.usize(0..n);
            if i != j {
                couplings.add(i, j, rng.f32() * 2.0 - 1.0);
            }
        }
        couplings.finalize();
        
        let mut cached = vec![0.0f32; n];
        let mut sweeper = MetropolisSweeper::new(1.0);
        
        // Warm up
        for _ in 0..5 {
            sweeper.sweep(&mut states, &biases, &couplings, &mut cached);
        }
        
        // Benchmark
        let start = Instant::now();
        let num_sweeps = 100;
        for _ in 0..num_sweeps {
            sweeper.sweep(&mut states, &biases, &couplings, &mut cached);
        }
        let elapsed = start.elapsed();
        
        let ns_per_sweep = elapsed.as_nanos() as f64 / num_sweeps as f64;
        let ns_per_spin = ns_per_sweep / n as f64;
        
        println!("10K pBits: {:.1}µs/sweep, {:.1}ns/spin", 
                 ns_per_sweep / 1000.0, ns_per_spin);
        
        // Should be under 500ns per spin in debug mode
        // (release mode is ~10x faster)
        assert!(ns_per_spin < 500.0, "Too slow: {:.1}ns/spin", ns_per_spin);
    }
}
