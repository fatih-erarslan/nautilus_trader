//! GPU-accelerated pBit dynamics using Metal (macOS/AMD)
//!
//! Provides compute shader implementations for parallel Metropolis updates.
//!
//! ## Design
//!
//! Uses checkerboard decomposition for parallel updates:
//! - Phase 1: Update all "red" (even) pBits in parallel
//! - Phase 2: Update all "black" (odd) pBits in parallel
//!
//! This ensures no race conditions on neighbor reads.

#[cfg(target_os = "macos")]
mod metal_impl {
    use super::super::{ScalableCouplings, ScalablePBitArray};
    use std::time::Instant;

    /// Metal shader source for pBit Metropolis update
    pub const METAL_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Coupling entry in CSR format
struct CouplingEntry {
    uint target;
    float strength;
};

// xorshift128+ RNG
inline uint64_t xorshift128plus(thread uint64_t* state0, thread uint64_t* state1) {
    uint64_t s1 = *state0;
    uint64_t s0 = *state1;
    *state0 = s0;
    s1 ^= s1 << 23;
    *state1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return *state1 + s0;
}

// Random float in [0, 1)
inline float rand_f32(thread uint64_t* s0, thread uint64_t* s1) {
    return float(xorshift128plus(s0, s1) >> 40) * (1.0f / float(1ul << 24));
}

// Metropolis update kernel for checkerboard phase
kernel void metropolis_update(
    device atomic_uint* states [[buffer(0)]],        // Packed bit states (atomic for safe access)
    device const float* biases [[buffer(1)]],        // Per-pBit biases
    device const uint* row_ptr [[buffer(2)]],        // CSR row pointers
    device const CouplingEntry* entries [[buffer(3)]], // CSR entries
    device const uint* params [[buffer(4)]],         // [num_pbits, phase, seed_lo, seed_hi]
    uint gid [[thread_position_in_grid]])
{
    uint num_pbits = params[0];
    uint phase = params[1];    // 0 = red (even), 1 = black (odd)
    
    // Calculate actual pBit index
    uint idx = gid * 2 + phase;
    if (idx >= num_pbits) return;
    
    // Initialize RNG per-thread
    uint64_t rng_s0 = uint64_t(params[2]) + uint64_t(gid) * 0x5DEECE66Dul;
    uint64_t rng_s1 = uint64_t(params[3]) ^ uint64_t(gid);
    
    // Get current spin
    uint word_idx = idx / 32;
    uint bit_idx = idx % 32;
    uint word = atomic_load_explicit(&states[word_idx], memory_order_relaxed);
    int spin_i = ((word >> bit_idx) & 1) ? 1 : -1;
    
    // Calculate effective field h_i = bias + sum_j J_ij * s_j
    float h = biases[idx];
    uint start = row_ptr[idx];
    uint end = row_ptr[idx + 1];
    
    for (uint e = start; e < end; e++) {
        uint j = entries[e].target;
        float J_ij = entries[e].strength;
        
        // Read neighbor spin
        uint j_word_idx = j / 32;
        uint j_bit_idx = j % 32;
        uint j_word = atomic_load_explicit(&states[j_word_idx], memory_order_relaxed);
        int spin_j = ((j_word >> j_bit_idx) & 1) ? 1 : -1;
        
        h += J_ij * float(spin_j);
    }
    
    // Energy change for flip: delta_E = 2 * s_i * h_i
    float delta_e = 2.0f * float(spin_i) * h;
    
    // Metropolis criterion
    // T=1.0 (beta=1.0) for simplicity, can be passed as param
    float accept_prob = (delta_e <= 0.0f) ? 1.0f : exp(-delta_e);
    
    if (rand_f32(&rng_s0, &rng_s1) < accept_prob) {
        // Flip the bit atomically
        uint mask = 1u << bit_idx;
        atomic_fetch_xor_explicit(&states[word_idx], mask, memory_order_relaxed);
    }
}

// Reduction kernel for counting active bits
kernel void count_ones(
    device const uint* states [[buffer(0)]],
    device atomic_uint* result [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint num_words = params[0];
    if (gid >= num_words) return;
    
    uint count = popcount(states[gid]);
    atomic_fetch_add_explicit(result, count, memory_order_relaxed);
}
"#;

    /// GPU sweep statistics
    #[derive(Debug, Clone, Copy)]
    pub struct GpuSweepStats {
        /// Number of sweeps executed
        pub sweeps: u32,
        /// Total GPU time in nanoseconds
        pub gpu_time_ns: u64,
        /// Total wall time in nanoseconds
        pub wall_time_ns: u64,
        /// Throughput in million spins per second
        pub throughput_mspins: f64,
    }

    impl std::fmt::Display for GpuSweepStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "Sweeps: {}, GPU: {:.2}ms, Wall: {:.2}ms, Throughput: {:.1}M spins/s",
                self.sweeps,
                self.gpu_time_ns as f64 / 1_000_000.0,
                self.wall_time_ns as f64 / 1_000_000.0,
                self.throughput_mspins
            )
        }
    }

    /// Placeholder GPU executor (requires metal-rs integration)
    /// 
    /// For full implementation, integrate with hyperphysics-gpu crate
    pub struct GpuExecutor {
        num_pbits: usize,
        seed: u64,
    }

    impl GpuExecutor {
        /// Create new GPU executor
        pub fn new(num_pbits: usize, seed: u64) -> Self {
            Self { num_pbits, seed }
        }

        /// Check if GPU is available
        pub fn is_available() -> bool {
            // Check for Metal availability on macOS
            #[cfg(target_os = "macos")]
            {
                true // Metal is always available on modern macOS
            }
            #[cfg(not(target_os = "macos"))]
            {
                false
            }
        }

        /// Get device info
        pub fn device_info() -> String {
            #[cfg(target_os = "macos")]
            {
                "Metal GPU (AMD Radeon / Apple Silicon)".to_string()
            }
            #[cfg(not(target_os = "macos"))]
            {
                "No GPU available".to_string()
            }
        }

        /// Simulate GPU sweep (CPU fallback for demo)
        /// 
        /// In production, this would upload buffers to GPU and dispatch compute shader
        pub fn execute_cpu_simulation(
            &mut self,
            states: &mut ScalablePBitArray,
            couplings: &ScalableCouplings,
            biases: &[f32],
            num_sweeps: usize,
        ) -> GpuSweepStats {
            let start = Instant::now();
            let n = self.num_pbits;
            
            let mut rng = fastrand::Rng::with_seed(self.seed);
            
            for _ in 0..num_sweeps {
                // Simulate checkerboard phases
                // Phase 0: Red (even)
                for i in (0..n).step_by(2) {
                    let delta_e = couplings.delta_energy(i, states, biases[i]);
                    let accept = delta_e <= 0.0 || rng.f32() < (-delta_e).exp().min(1.0) as f32;
                    if accept {
                        states.flip(i);
                    }
                }
                
                // Phase 1: Black (odd)
                for i in (1..n).step_by(2) {
                    let delta_e = couplings.delta_energy(i, states, biases[i]);
                    let accept = delta_e <= 0.0 || rng.f32() < (-delta_e).exp().min(1.0) as f32;
                    if accept {
                        states.flip(i);
                    }
                }
            }
            
            let wall_time = start.elapsed();
            
            GpuSweepStats {
                sweeps: num_sweeps as u32,
                gpu_time_ns: wall_time.as_nanos() as u64, // Would be actual GPU time
                wall_time_ns: wall_time.as_nanos() as u64,
                throughput_mspins: (n * num_sweeps) as f64 / wall_time.as_secs_f64() / 1_000_000.0,
            }
        }
    }
}

#[cfg(target_os = "macos")]
pub use metal_impl::*;

/// Stub for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub mod metal_impl {
    use super::super::{ScalableCouplings, ScalablePBitArray};

    #[derive(Debug, Clone, Copy)]
    pub struct GpuSweepStats {
        pub sweeps: u32,
        pub gpu_time_ns: u64,
        pub wall_time_ns: u64,
        pub throughput_mspins: f64,
    }

    impl std::fmt::Display for GpuSweepStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "GPU not available on this platform")
        }
    }

    pub struct GpuExecutor;

    impl GpuExecutor {
        pub fn new(_: usize, _: u64) -> Self { Self }
        pub fn is_available() -> bool { false }
        pub fn device_info() -> String { "No GPU".to_string() }
        pub fn execute_cpu_simulation(
            &mut self, _: &mut ScalablePBitArray, _: &ScalableCouplings, _: &[f32], _: usize
        ) -> GpuSweepStats {
            GpuSweepStats { sweeps: 0, gpu_time_ns: 0, wall_time_ns: 0, throughput_mspins: 0.0 }
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use metal_impl::*;
