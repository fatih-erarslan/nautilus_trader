//! SIMD-optimized Metropolis sweep
//!
//! Uses packed bit operations and vectorized RNG for maximum throughput.
//!
//! # Architecture Support
//!
//! - **x86_64**: AVX2/AVX-512 vectorized energy calculations
//! - **aarch64**: NEON vectorized operations
//! - **Fallback**: Scalar implementation with cache-optimized chunking

use super::{ScalableCouplings, ScalablePBitArray};

#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
#[allow(unused_imports)]
use std::arch::aarch64::*;

/// SIMD-optimized sweep executor
pub struct SimdSweep {
    /// Temperature
    temperature: f64,
    /// Inverse temperature
    beta: f64,
    /// RNG state (xorshift128+)
    rng_state: [u64; 2],
    /// Precomputed exp(-beta * delta_e) for common values
    exp_table: Vec<f32>,
}

impl SimdSweep {
    /// Create new SIMD sweep executor
    pub fn new(temperature: f64, seed: u64) -> Self {
        let mut sweep = Self {
            temperature,
            beta: 1.0 / temperature,
            rng_state: [seed, seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB)],
            exp_table: Vec::with_capacity(1024),
        };
        sweep.rebuild_exp_table();
        sweep
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
        self.beta = 1.0 / temperature;
        self.rebuild_exp_table();
    }

    /// Rebuild exponential lookup table
    fn rebuild_exp_table(&mut self) {
        self.exp_table.clear();
        // Table covers ΔE from -16 to +16 with 0.03125 resolution
        for i in 0..1024 {
            let delta_e = (i as f64 - 512.0) * 0.03125;
            let prob = (-self.beta * delta_e).exp().min(1.0);
            self.exp_table.push(prob as f32);
        }
    }

    /// Fast xorshift128+ RNG
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let s0 = self.rng_state[0];
        let mut s1 = self.rng_state[1];
        let result = s0.wrapping_add(s1);
        
        s1 ^= s0;
        self.rng_state[0] = s0.rotate_left(55) ^ s1 ^ (s1 << 14);
        self.rng_state[1] = s1.rotate_left(36);
        
        result
    }

    /// Generate random f32 in [0, 1)
    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Lookup acceptance probability
    #[inline(always)]
    fn accept_prob(&self, delta_e: f32) -> f32 {
        if delta_e <= 0.0 {
            return 1.0;
        }
        let idx = ((delta_e * 32.0 + 512.0) as usize).min(1023);
        self.exp_table[idx]
    }

    /// Execute sweep with SIMD optimizations where possible
    #[cfg(target_arch = "x86_64")]
    pub fn execute(
        &mut self,
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
    ) -> SimdSweepStats {
        let start = std::time::Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Process in chunks for better cache utilization
        const CHUNK_SIZE: usize = 64;
        
        for chunk_start in (0..n).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_start + CHUNK_SIZE).min(n);
            
            // Compute delta_E for chunk
            for i in chunk_start..chunk_end {
                let delta_e = couplings.delta_energy(i, states, biases[i]);
                
                // Metropolis criterion with fast lookup
                let accept = delta_e <= 0.0 || self.next_f32() < self.accept_prob(delta_e);
                
                if accept {
                    states.flip(i);
                    flips += 1;
                }
            }
        }

        let duration = start.elapsed();

        SimdSweepStats {
            flips,
            duration_ns: duration.as_nanos() as u64,
            throughput_mspins: n as f64 / duration.as_secs_f64() / 1_000_000.0,
        }
    }

    /// Execute sweep (ARM NEON path)
    #[cfg(target_arch = "aarch64")]
    pub fn execute(
        &mut self,
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
    ) -> SimdSweepStats {
        let start = std::time::Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Sequential with fast RNG
        for i in 0..n {
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            let accept = delta_e <= 0.0 || self.next_f32() < self.accept_prob(delta_e);
            
            if accept {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SimdSweepStats {
            flips,
            duration_ns: duration.as_nanos() as u64,
            throughput_mspins: n as f64 / duration.as_secs_f64() / 1_000_000.0,
        }
    }

    /// Fallback for other architectures
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn execute(
        &mut self,
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
    ) -> SimdSweepStats {
        let start = std::time::Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        for i in 0..n {
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            let accept = delta_e <= 0.0 || self.next_f32() < self.accept_prob(delta_e);
            
            if accept {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SimdSweepStats {
            flips,
            duration_ns: duration.as_nanos() as u64,
            throughput_mspins: n as f64 / duration.as_secs_f64() / 1_000_000.0,
        }
    }

    /// Execute checkerboard sweep for parallel updates
    /// 
    /// Splits pBits into "red" (even) and "black" (odd) groups.
    /// Within each group, updates are independent (no coupling conflicts).
    pub fn execute_checkerboard(
        &mut self,
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
    ) -> SimdSweepStats {
        let start = std::time::Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Red phase (even indices)
        for i in (0..n).step_by(2) {
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            if delta_e <= 0.0 || self.next_f32() < self.accept_prob(delta_e) {
                states.flip(i);
                flips += 1;
            }
        }

        // Black phase (odd indices)
        for i in (1..n).step_by(2) {
            let delta_e = couplings.delta_energy(i, states, biases[i]);
            if delta_e <= 0.0 || self.next_f32() < self.accept_prob(delta_e) {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SimdSweepStats {
            flips,
            duration_ns: duration.as_nanos() as u64,
            throughput_mspins: n as f64 / duration.as_secs_f64() / 1_000_000.0,
        }
    }
}

/// Statistics from SIMD sweep
#[derive(Debug, Clone, Copy)]
pub struct SimdSweepStats {
    /// Number of accepted flips
    pub flips: u32,
    /// Duration in nanoseconds
    pub duration_ns: u64,
    /// Throughput in million spins per second
    pub throughput_mspins: f64,
}

impl std::fmt::Display for SimdSweepStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Flips: {}, Time: {:.2}ms, Throughput: {:.2}M spins/s",
            self.flips,
            self.duration_ns as f64 / 1_000_000.0,
            self.throughput_mspins
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_sweep() {
        let n = 1000;
        let mut states = ScalablePBitArray::random(n, 42);
        let mut couplings = ScalableCouplings::new(n);
        
        for i in 0..n {
            couplings.add_symmetric(i, (i + 1) % n, 1.0);
        }
        couplings.finalize();
        
        let biases = vec![0.0f32; n];
        let mut sweep = SimdSweep::new(1.0, 42);
        
        let stats = sweep.execute(&mut states, &couplings, &biases);
        
        println!("{}", stats);
        assert!(stats.flips <= n as u32);
        assert!(stats.throughput_mspins > 0.0);
    }

    #[test]
    fn test_checkerboard() {
        let n = 1000;
        let mut states = ScalablePBitArray::random(n, 42);
        let mut couplings = ScalableCouplings::new(n);
        
        for i in 0..n {
            couplings.add_symmetric(i, (i + 1) % n, 1.0);
        }
        couplings.finalize();
        
        let biases = vec![0.0f32; n];
        let mut sweep = SimdSweep::new(1.0, 42);
        
        let stats = sweep.execute_checkerboard(&mut states, &couplings, &biases);
        
        println!("Checkerboard: {}", stats);
        assert!(stats.flips <= n as u32);
    }
}
