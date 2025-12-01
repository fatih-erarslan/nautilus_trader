//! # Scalable pBit Fabric
//!
//! Ultra-high-performance pBit implementation designed for billion-scale systems.
//! Inspired by ruvector's scalability techniques.
//!
//! ## Key Optimizations
//!
//! 1. **Packed Bit Storage**: 64 pBits per u64 word
//! 2. **Structure-of-Arrays**: Cache-friendly memory layout
//! 3. **Sparse CSR Couplings**: O(E) instead of O(N²)
//! 4. **Lock-Free Updates**: No contention on parallel access
//! 5. **SIMD Acceleration**: AVX2/AVX-512 for batch operations
//! 6. **Checkerboard Parallelism**: GPU-ready parallel sweeps
//!
//! ## Performance Targets
//!
//! | Scale | Sweep Latency | Memory |
//! |-------|---------------|--------|
//! | 64 pBits | <100ns | 8 bytes |
//! | 1K pBits | <1µs | 128 bytes |
//! | 64K pBits | <50µs | 8 KB |
//! | 1M pBits | <1ms | 128 KB |
//! | 1B pBits | <100ms (GPU) | 125 MB |

mod packed_array;
mod sparse_couplings;
mod metropolis;

pub use packed_array::{PackedPBitArray, BITS_PER_WORD};
pub use sparse_couplings::{SparseCouplings, CouplingEntry};
pub use metropolis::{MetropolisSweeper, SweepResult};

/// Configuration for scalable pBit fabric
#[derive(Debug, Clone)]
pub struct ScalablePBitConfig {
    /// Number of pBits
    pub num_pbits: usize,
    /// Average coupling degree (connections per pBit)
    pub avg_degree: usize,
    /// Default temperature
    pub temperature: f64,
    /// Enable SIMD optimizations
    pub use_simd: bool,
    /// Pre-allocate for this many couplings
    pub coupling_capacity: Option<usize>,
}

impl Default for ScalablePBitConfig {
    fn default() -> Self {
        Self {
            num_pbits: 1024,
            avg_degree: 6,
            temperature: 1.0,
            use_simd: true,
            coupling_capacity: None,
        }
    }
}

impl ScalablePBitConfig {
    /// Configuration for 64K pBits (fits in L2 cache)
    pub fn l2_optimal() -> Self {
        Self {
            num_pbits: 65536,
            avg_degree: 8,
            temperature: 1.0,
            use_simd: true,
            coupling_capacity: Some(65536 * 8),
        }
    }

    /// Configuration for 1M pBits
    pub fn million() -> Self {
        Self {
            num_pbits: 1_000_000,
            avg_degree: 10,
            temperature: 1.0,
            use_simd: true,
            coupling_capacity: Some(10_000_000),
        }
    }
}

/// High-performance scalable pBit fabric
pub struct ScalablePBitFabric {
    /// Packed bit states (64 pBits per word)
    states: PackedPBitArray,
    /// Per-pBit biases
    biases: Vec<f32>,
    /// Sparse coupling matrix (CSR format)
    couplings: SparseCouplings,
    /// Metropolis sweeper
    sweeper: MetropolisSweeper,
    /// Configuration
    config: ScalablePBitConfig,
    /// Cached effective fields (for incremental updates)
    cached_fields: Vec<f32>,
}

impl ScalablePBitFabric {
    /// Create a new scalable pBit fabric
    pub fn new(config: ScalablePBitConfig) -> Self {
        let states = PackedPBitArray::new(config.num_pbits);
        let biases = vec![0.0f32; config.num_pbits];
        let coupling_capacity = config.coupling_capacity
            .unwrap_or(config.num_pbits * config.avg_degree);
        let couplings = SparseCouplings::with_capacity(config.num_pbits, coupling_capacity);
        let sweeper = MetropolisSweeper::new(config.temperature);
        let cached_fields = vec![0.0f32; config.num_pbits];

        Self {
            states,
            biases,
            couplings,
            sweeper,
            config,
            cached_fields,
        }
    }

    /// Create with random sparse couplings (Erdős–Rényi style)
    pub fn with_random_couplings(config: ScalablePBitConfig, seed: u64) -> Self {
        let mut fabric = Self::new(config.clone());
        
        // Generate random sparse couplings
        let mut rng = fastrand::Rng::with_seed(seed);
        let n = config.num_pbits;
        let target_edges = n * config.avg_degree / 2; // Undirected, so /2
        
        for _ in 0..target_edges {
            let i = rng.usize(0..n);
            let j = rng.usize(0..n);
            if i != j {
                let strength = rng.f32() * 2.0 - 1.0; // [-1, 1]
                fabric.add_coupling(i, j, strength);
            }
        }
        
        fabric.couplings.finalize();
        fabric
    }

    /// Add a coupling between two pBits
    #[inline]
    pub fn add_coupling(&mut self, i: usize, j: usize, strength: f32) {
        self.couplings.add(i, j, strength);
        // Symmetric coupling
        self.couplings.add(j, i, strength);
    }

    /// Set bias for a specific pBit
    #[inline]
    pub fn set_bias(&mut self, idx: usize, bias: f32) {
        self.biases[idx] = bias;
    }

    /// Perform a single Metropolis sweep
    ///
    /// Returns the number of flips and sweep duration
    #[inline]
    pub fn metropolis_sweep(&mut self) -> SweepResult {
        self.sweeper.sweep(
            &mut self.states,
            &self.biases,
            &self.couplings,
            &mut self.cached_fields,
        )
    }

    /// Perform multiple sweeps
    pub fn simulate(&mut self, num_sweeps: usize) -> Vec<SweepResult> {
        (0..num_sweeps)
            .map(|_| self.metropolis_sweep())
            .collect()
    }

    /// Get current state of a pBit
    #[inline]
    pub fn get_state(&self, idx: usize) -> bool {
        self.states.get(idx)
    }

    /// Set state of a pBit
    #[inline]
    pub fn set_state(&mut self, idx: usize, state: bool) {
        self.states.set(idx, state);
    }

    /// Flip a pBit
    #[inline]
    pub fn flip(&mut self, idx: usize) {
        self.states.flip(idx);
    }

    /// Get all states as a vector (for compatibility)
    pub fn states_vec(&self) -> Vec<bool> {
        (0..self.config.num_pbits)
            .map(|i| self.states.get(i))
            .collect()
    }

    /// Count active pBits (state = 1)
    pub fn count_active(&self) -> usize {
        self.states.count_ones()
    }

    /// Calculate magnetization (mean spin)
    pub fn magnetization(&self) -> f64 {
        let ones = self.count_active();
        let zeros = self.config.num_pbits - ones;
        (ones as f64 - zeros as f64) / self.config.num_pbits as f64
    }

    /// Calculate total energy
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0f64;
        
        // Bias contribution: -Σ h_i s_i
        for i in 0..self.config.num_pbits {
            let s_i = if self.states.get(i) { 1.0 } else { -1.0 };
            energy -= self.biases[i] as f64 * s_i;
        }
        
        // Coupling contribution: -Σ J_ij s_i s_j
        for i in 0..self.config.num_pbits {
            let s_i = if self.states.get(i) { 1.0 } else { -1.0 };
            for (j, j_ij) in self.couplings.neighbors(i) {
                let s_j = if self.states.get(j) { 1.0 } else { -1.0 };
                energy -= 0.5 * j_ij as f64 * s_i * s_j; // 0.5 for double counting
            }
        }
        
        energy
    }

    /// Number of pBits
    #[inline]
    pub fn len(&self) -> usize {
        self.config.num_pbits
    }

    /// Set temperature
    #[inline]
    pub fn set_temperature(&mut self, temperature: f64) {
        self.sweeper.set_temperature(temperature);
    }

    /// Get configuration
    pub fn config(&self) -> &ScalablePBitConfig {
        &self.config
    }

    /// Get coupling statistics
    pub fn coupling_stats(&self) -> (usize, f64) {
        let nnz = self.couplings.nnz();
        let avg_degree = nnz as f64 / self.config.num_pbits as f64;
        (nnz, avg_degree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalable_pbit_creation() {
        let config = ScalablePBitConfig::default();
        let fabric = ScalablePBitFabric::new(config);
        assert_eq!(fabric.len(), 1024);
    }

    #[test]
    fn test_random_couplings() {
        let config = ScalablePBitConfig {
            num_pbits: 1000,
            avg_degree: 6,
            ..Default::default()
        };
        let fabric = ScalablePBitFabric::with_random_couplings(config, 42);
        
        let (nnz, avg_degree) = fabric.coupling_stats();
        assert!(nnz > 0);
        assert!(avg_degree > 1.0);
    }

    #[test]
    fn test_metropolis_sweep() {
        let config = ScalablePBitConfig {
            num_pbits: 1000,
            avg_degree: 6,
            temperature: 1.0,
            ..Default::default()
        };
        let mut fabric = ScalablePBitFabric::with_random_couplings(config, 42);
        
        let result = fabric.metropolis_sweep();
        assert!(result.duration_ns > 0);
    }

    #[test]
    fn test_64k_pbit_performance() {
        let config = ScalablePBitConfig::l2_optimal();
        let mut fabric = ScalablePBitFabric::with_random_couplings(config, 12345);
        
        // Warm up
        for _ in 0..10 {
            fabric.metropolis_sweep();
        }
        
        // Benchmark
        let start = std::time::Instant::now();
        let num_sweeps = 100;
        for _ in 0..num_sweeps {
            fabric.metropolis_sweep();
        }
        let elapsed = start.elapsed();
        
        let us_per_sweep = elapsed.as_micros() as f64 / num_sweeps as f64;
        let ns_per_spin = us_per_sweep * 1000.0 / 65536.0;
        println!("64K pBits: {:.2}µs per sweep, {:.1}ns/spin", us_per_sweep, ns_per_spin);
        
        // Should be under 500ns per spin in debug mode
        assert!(ns_per_spin < 500.0, "Sweep too slow: {:.1}ns/spin", ns_per_spin);
    }
}
