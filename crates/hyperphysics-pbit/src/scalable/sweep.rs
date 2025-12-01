//! Metropolis-Hastings sweep implementation
//!
//! Optimized for minimum latency with no heap allocation in hot path.

use super::{ScalableCouplings, ScalablePBitArray};
use std::time::Instant;

/// Statistics from a single sweep
#[derive(Debug, Clone, Copy)]
pub struct SweepStats {
    /// Number of accepted flips
    pub flips: u32,
    /// Duration in nanoseconds
    pub duration_ns: u64,
    /// Acceptance rate
    pub acceptance_rate: f32,
}

/// Metropolis-Hastings sweep executor
///
/// Maintains RNG state and precomputed acceptance table.
pub struct MetropolisSweep {
    /// Temperature (kT in natural units)
    temperature: f64,
    /// Inverse temperature
    beta: f64,
    /// Fast RNG
    rng: fastrand::Rng,
    /// Precomputed acceptance probabilities for common ΔE values
    /// Index = (ΔE * 16 + 128), clamped to [0, 255]
    acceptance_table: [f32; 256],
}

impl MetropolisSweep {
    /// Create new sweep executor
    pub fn new(temperature: f64, seed: u64) -> Self {
        let mut sweep = Self {
            temperature,
            beta: 1.0 / temperature,
            rng: fastrand::Rng::with_seed(seed),
            acceptance_table: [0.0; 256],
        };
        sweep.rebuild_table();
        sweep
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
        self.beta = 1.0 / temperature;
        self.rebuild_table();
    }

    /// Get temperature
    #[inline]
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Rebuild acceptance probability table
    fn rebuild_table(&mut self) {
        for i in 0..256 {
            let delta_e = (i as f64 - 128.0) * 0.0625; // Range: [-8, +8]
            let prob = (-self.beta * delta_e).exp().min(1.0);
            self.acceptance_table[i] = prob as f32;
        }
    }

    /// Fast acceptance probability lookup
    #[inline(always)]
    fn accept_prob(&self, delta_e: f32) -> f32 {
        if delta_e <= 0.0 {
            return 1.0;
        }
        let idx = ((delta_e * 16.0 + 128.0) as usize).min(255);
        self.acceptance_table[idx]
    }

    /// Execute one full sweep (visit each pBit once)
    ///
    /// Uses sequential order for cache efficiency.
    pub fn execute(
        &mut self,
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
    ) -> SweepStats {
        let start = Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Sequential sweep for cache efficiency
        for i in 0..n {
            let delta_e = couplings.delta_energy(i, states, biases[i]);

            // Metropolis criterion
            let accept = delta_e <= 0.0 || self.rng.f32() < self.accept_prob(delta_e);

            if accept {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SweepStats {
            flips,
            duration_ns: duration.as_nanos() as u64,
            acceptance_rate: flips as f32 / n as f32,
        }
    }

    /// Execute sweep with random visit order
    ///
    /// More accurate for physics but slower due to random access.
    pub fn execute_random(
        &mut self,
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
    ) -> SweepStats {
        let start = Instant::now();
        let n = states.len();
        let mut flips = 0u32;

        // Random order: pick N random pBits
        for _ in 0..n {
            let i = self.rng.usize(0..n);
            let delta_e = couplings.delta_energy(i, states, biases[i]);

            if delta_e <= 0.0 || self.rng.f32() < self.accept_prob(delta_e) {
                states.flip(i);
                flips += 1;
            }
        }

        let duration = start.elapsed();

        SweepStats {
            flips,
            duration_ns: duration.as_nanos() as u64,
            acceptance_rate: flips as f32 / n as f32,
        }
    }

    /// Execute multiple sweeps
    pub fn run(&mut self, 
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
        num_sweeps: usize,
    ) -> Vec<SweepStats> {
        (0..num_sweeps)
            .map(|_| self.execute(states, couplings, biases))
            .collect()
    }

    /// Execute sweeps and return aggregate stats
    pub fn run_aggregate(
        &mut self,
        states: &mut ScalablePBitArray,
        couplings: &ScalableCouplings,
        biases: &[f32],
        num_sweeps: usize,
    ) -> AggregateStats {
        let start = Instant::now();
        let mut total_flips = 0u64;

        for _ in 0..num_sweeps {
            let stats = self.execute(states, couplings, biases);
            total_flips += stats.flips as u64;
        }

        let duration = start.elapsed();

        AggregateStats {
            num_sweeps,
            total_flips,
            total_duration_ns: duration.as_nanos() as u64,
            ns_per_sweep: duration.as_nanos() as u64 / num_sweeps as u64,
            ns_per_spin: duration.as_nanos() as u64 / (num_sweeps * states.len()) as u64,
            avg_acceptance: total_flips as f64 / (num_sweeps * states.len()) as f64,
            final_magnetization: states.magnetization(),
        }
    }
}

/// Aggregate statistics from multiple sweeps
#[derive(Debug, Clone)]
pub struct AggregateStats {
    /// Number of sweeps executed
    pub num_sweeps: usize,
    /// Total flips across all sweeps
    pub total_flips: u64,
    /// Total duration in nanoseconds
    pub total_duration_ns: u64,
    /// Nanoseconds per sweep
    pub ns_per_sweep: u64,
    /// Nanoseconds per spin update
    pub ns_per_spin: u64,
    /// Average acceptance rate
    pub avg_acceptance: f64,
    /// Final magnetization
    pub final_magnetization: f64,
}

impl std::fmt::Display for AggregateStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Sweeps: {}, Time: {:.2}ms, {:.1}ns/spin, Accept: {:.1}%, Mag: {:.3}",
            self.num_sweeps,
            self.total_duration_ns as f64 / 1_000_000.0,
            self.ns_per_spin,
            self.avg_acceptance * 100.0,
            self.final_magnetization
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ring_system(n: usize) -> (ScalablePBitArray, ScalableCouplings, Vec<f32>) {
        let states = ScalablePBitArray::random(n, 42);
        let mut couplings = ScalableCouplings::new(n);
        
        for i in 0..n {
            couplings.add_symmetric(i, (i + 1) % n, 1.0);
        }
        couplings.finalize();
        
        let biases = vec![0.0f32; n];
        (states, couplings, biases)
    }

    #[test]
    fn test_single_sweep() {
        let (mut states, couplings, biases) = make_ring_system(100);
        let mut sweep = MetropolisSweep::new(1.0, 42);
        
        let stats = sweep.execute(&mut states, &couplings, &biases);
        
        assert!(stats.flips <= 100);
        assert!(stats.duration_ns > 0);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_temperature_effect() {
        let n = 500;
        
        // Hot system
        let (mut states_hot, couplings, biases) = make_ring_system(n);
        let mut sweep_hot = MetropolisSweep::new(10.0, 42);
        let stats_hot = sweep_hot.run_aggregate(&mut states_hot, &couplings, &biases, 100);
        
        // Cold system
        let (mut states_cold, couplings, biases) = make_ring_system(n);
        let mut sweep_cold = MetropolisSweep::new(0.1, 43);
        let stats_cold = sweep_cold.run_aggregate(&mut states_cold, &couplings, &biases, 100);
        
        println!("Hot (T=10): {}", stats_hot);
        println!("Cold (T=0.1): {}", stats_cold);
        
        // Hot should have higher acceptance rate
        assert!(
            stats_hot.avg_acceptance > stats_cold.avg_acceptance,
            "Hot should accept more: hot={:.3}, cold={:.3}",
            stats_hot.avg_acceptance,
            stats_cold.avg_acceptance
        );
    }

    #[test]
    fn test_aggregate() {
        let (mut states, couplings, biases) = make_ring_system(1000);
        let mut sweep = MetropolisSweep::new(1.0, 42);
        
        let stats = sweep.run_aggregate(&mut states, &couplings, &biases, 100);
        
        println!("{}", stats);
        
        assert_eq!(stats.num_sweeps, 100);
        assert!(stats.ns_per_spin > 0);
    }
}
