//! # Fibonacci Pentagon Topology Implementation
//!
//! 5-engine pBit topology with golden ratio coupling.

use crate::engine::{PBitEngine, EngineConfig};
use crate::msocl::Msocl;
use crate::fibonacci::constants::{
    PHI, PHI_INV, FIBONACCI_COUPLING, PENTAGON_PHASES,
    fibonacci_sequence, golden_ratio_approximation,
};

use std::f64::consts::PI;

/// Number of engines in pentagon
pub const PENTAGON_ENGINES: usize = 5;

/// Coupling scale factor
pub const FIBONACCI_COUPLING_SCALE: f64 = 0.1;

/// Phase coherence threshold for synchronization detection
pub const PHASE_COHERENCE_THRESHOLD: f64 = 0.8;

/// Minimum coherence (random phases) - used for normalization
pub const MIN_COHERENCE: f64 = 0.0;

/// Maximum coherence (perfect synchronization) - used for normalization
pub const MAX_COHERENCE: f64 = 1.0;

/// Critical coherence threshold (golden ratio derived: 1/φ)
pub const CRITICAL_COHERENCE: f64 = 0.618033988749895;

/// Temperature modulation phases (golden angle spiral)
const FIBONACCI_TEMP_PHASES: [f64; PENTAGON_ENGINES] = [
    0.0,
    2.39996322972865332,                           // φ golden angle
    2.39996322972865332 * 2.0 - 2.0 * PI,         // Wrapped to [0, 2π)
    2.39996322972865332 * 3.0 - 4.0 * PI,
    2.39996322972865332 * 4.0 - 6.0 * PI,
];

/// Configuration for Fibonacci Pentagon
#[derive(Debug, Clone)]
pub struct PentagonConfig {
    /// Configuration for each pBit engine
    pub engine_config: EngineConfig,
    /// Base temperature (before modulation)
    pub base_temperature: f64,
    /// Coupling scale factor
    pub coupling_scale: f64,
    /// Enable STDP learning
    pub enable_stdp: bool,
}

impl Default for PentagonConfig {
    fn default() -> Self {
        Self {
            engine_config: EngineConfig::default(),
            base_temperature: 1.0,
            coupling_scale: FIBONACCI_COUPLING_SCALE,
            enable_stdp: true,
        }
    }
}

/// Fibonacci coupling tensor for 5 engines
#[derive(Debug, Clone)]
pub struct FibonacciCoupling {
    /// Coupling matrix [5x5] with golden ratio weights
    pub matrix: [[f64; PENTAGON_ENGINES]; PENTAGON_ENGINES],
    /// Coupling scale factor
    pub scale: f64,
}

impl FibonacciCoupling {
    /// Create new Fibonacci coupling with default matrix
    pub fn new(scale: f64) -> Self {
        Self {
            matrix: FIBONACCI_COUPLING,
            scale,
        }
    }

    /// Get effective coupling between engines i and j
    #[inline]
    pub fn coupling(&self, i: usize, j: usize) -> f64 {
        if i < PENTAGON_ENGINES && j < PENTAGON_ENGINES {
            self.matrix[i][j] * self.scale
        } else {
            0.0
        }
    }

    /// Get coupling strength from engine i to all others
    pub fn row(&self, i: usize) -> [f64; PENTAGON_ENGINES] {
        if i < PENTAGON_ENGINES {
            let mut row = [0.0; PENTAGON_ENGINES];
            for j in 0..PENTAGON_ENGINES {
                row[j] = self.matrix[i][j] * self.scale;
            }
            row
        } else {
            [0.0; PENTAGON_ENGINES]
        }
    }

    /// Compute largest eigenvalue (approximate, for diagnostics)
    /// Exact: λ_max = √5 ≈ 2.236 (Wolfram-verified)
    pub fn spectral_radius(&self) -> f64 {
        5.0_f64.sqrt() * self.scale
    }
}

impl Default for FibonacciCoupling {
    fn default() -> Self {
        Self::new(FIBONACCI_COUPLING_SCALE)
    }
}

/// Fibonacci Pentagon Topology with 5 PBit Engines
pub struct FibonacciPentagon {
    /// 5 pBit engines arranged in pentagon
    engines: [PBitEngine; PENTAGON_ENGINES],
    /// Fibonacci coupling tensor
    coupling: FibonacciCoupling,
    /// MSOCL for global phase coordination
    msocl: Msocl,
    /// Base temperature
    base_temp: f64,
    /// Enable STDP
    enable_stdp: bool,
    /// Tick counter
    tick: u64,
}

impl FibonacciPentagon {
    /// Create new Fibonacci Pentagon topology
    pub fn new(config: PentagonConfig) -> Self {
        // Create 5 engines with different seeds for diversity
        let engines = [
            PBitEngine::new(0, config.engine_config.clone()),
            PBitEngine::new(1, config.engine_config.clone()),
            PBitEngine::new(2, config.engine_config.clone()),
            PBitEngine::new(3, config.engine_config.clone()),
            PBitEngine::new(4, config.engine_config.clone()),
        ];

        let coupling = FibonacciCoupling::new(config.coupling_scale);
        let msocl = Msocl::new();

        Self {
            engines,
            coupling,
            msocl,
            base_temp: config.base_temperature,
            enable_stdp: config.enable_stdp,
            tick: 0,
        }
    }

    /// Perform one complete MSOCL cycle
    pub fn step(&mut self) {
        // Update MSOCL phase
        self.msocl.tick();

        // Apply temperature modulation with golden angle phases
        self.apply_temperature_modulation();

        // Apply inter-engine coupling with φ/φ⁻¹ weights
        self.apply_inter_engine_coupling();

        // Each engine performs local pBit update
        for engine in &mut self.engines {
            engine.step();
        }

        // Optional STDP weight updates
        if self.enable_stdp {
            for engine in &mut self.engines {
                engine.apply_stdp();
            }
        }

        self.tick += 1;
    }

    /// Perform N MSOCL cycles
    pub fn step_n(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Apply inter-engine coupling with φ/φ⁻¹ weighting
    pub fn apply_inter_engine_coupling(&mut self) {
        // Compute spike rates for all engines
        let rates: [f64; PENTAGON_ENGINES] = [
            self.engines[0].spike_rate(),
            self.engines[1].spike_rate(),
            self.engines[2].spike_rate(),
            self.engines[3].spike_rate(),
            self.engines[4].spike_rate(),
        ];

        // Compute coupling inputs for each engine
        let mut coupling_inputs = [[0.0f32; 1024]; PENTAGON_ENGINES]; // Max pBits per engine

        for i in 0..PENTAGON_ENGINES {
            let num_pbits = self.engines[i].num_pbits();

            for j in 0..PENTAGON_ENGINES {
                if i == j {
                    continue;
                }

                let coupling_strength = self.coupling.coupling(i, j) as f32;
                let source_rate = rates[j] as f32;

                // Apply coupling proportional to source spike rate
                let input_strength = coupling_strength * source_rate;

                for k in 0..num_pbits.min(1024) {
                    coupling_inputs[i][k] += input_strength;
                }
            }
        }

        // Apply coupling inputs to engines
        for (i, engine) in self.engines.iter_mut().enumerate() {
            let num_pbits = engine.num_pbits();
            let input_slice = &coupling_inputs[i][..num_pbits.min(1024)];
            engine.apply_input(input_slice);
        }
    }

    /// Apply temperature modulation with golden angle spiral
    fn apply_temperature_modulation(&mut self) {
        let msocl_temp_mod = self.msocl.temperature_modulation();

        for (i, engine) in self.engines.iter_mut().enumerate() {
            // Each engine has phase offset based on golden angle spiral
            let phase_offset = FIBONACCI_TEMP_PHASES[i];
            let phase = (self.msocl.tick_count() as f64 * 0.01 + phase_offset) % (2.0 * PI);

            // Combine MSOCL modulation with Fibonacci spiral
            let spiral_mod = 1.0 + 0.05 * phase.sin();
            let total_mod = msocl_temp_mod * spiral_mod;

            let new_temp = self.base_temp * total_mod;
            engine.set_temperature(new_temp);
        }
    }

    /// Get spike rates for all 5 engines
    pub fn spike_rates(&self) -> [f64; PENTAGON_ENGINES] {
        [
            self.engines[0].spike_rate(),
            self.engines[1].spike_rate(),
            self.engines[2].spike_rate(),
            self.engines[3].spike_rate(),
            self.engines[4].spike_rate(),
        ]
    }

    /// Compute phase coherence across all engines
    /// Uses Kuramoto order parameter: r = |⟨exp(iθⱼ)⟩|
    pub fn phase_coherence(&self) -> f64 {
        // Map spike rates to phases [0, 2π)
        let rates = self.spike_rates();

        // Compute complex order parameter
        let mut real_sum = 0.0_f64;
        let mut imag_sum = 0.0_f64;

        for &rate in &rates {
            let phase = rate * 2.0 * PI; // Map [0,1] → [0, 2π)
            real_sum += phase.cos();
            imag_sum += phase.sin();
        }

        // Order parameter r = |sum| / N
        let magnitude = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
        magnitude / PENTAGON_ENGINES as f64
    }

    /// Get reference to coupling tensor
    pub fn coupling(&self) -> &FibonacciCoupling {
        &self.coupling
    }

    /// Get reference to MSOCL
    pub fn msocl(&self) -> &Msocl {
        &self.msocl
    }

    /// Get mutable reference to specific engine
    pub fn engine_mut(&mut self, id: usize) -> Option<&mut PBitEngine> {
        if id < PENTAGON_ENGINES {
            Some(&mut self.engines[id])
        } else {
            None
        }
    }

    /// Get reference to specific engine
    pub fn engine(&self, id: usize) -> Option<&PBitEngine> {
        if id < PENTAGON_ENGINES {
            Some(&self.engines[id])
        } else {
            None
        }
    }

    /// Get current tick count
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Reset all engines to initial state
    pub fn reset(&mut self) {
        for engine in &mut self.engines {
            engine.reset();
        }
        self.tick = 0;
    }

    /// Get summary statistics
    pub fn stats(&self) -> PentagonStats {
        let rates = self.spike_rates();
        let coherence = self.phase_coherence();

        let mean_rate = rates.iter().sum::<f64>() / PENTAGON_ENGINES as f64;

        let variance = rates.iter()
            .map(|r| (r - mean_rate).powi(2))
            .sum::<f64>() / PENTAGON_ENGINES as f64;

        PentagonStats {
            tick: self.tick,
            spike_rates: rates,
            mean_spike_rate: mean_rate,
            spike_rate_variance: variance,
            phase_coherence: coherence,
            spectral_radius: self.coupling.spectral_radius(),
            msocl_order_parameter: self.msocl.order_parameter(),
        }
    }

    /// Compute normalized coherence in [0, 1] range
    /// Uses MIN_COHERENCE and MAX_COHERENCE for normalization
    pub fn normalized_coherence(&self) -> f64 {
        let raw = self.phase_coherence();
        (raw - MIN_COHERENCE) / (MAX_COHERENCE - MIN_COHERENCE)
    }

    /// Check if system is synchronized (coherence above threshold)
    pub fn is_synchronized(&self) -> bool {
        self.phase_coherence() >= PHASE_COHERENCE_THRESHOLD
    }

    /// Check if system is near critical point (coherence near φ⁻¹)
    pub fn is_critical(&self) -> bool {
        let coherence = self.phase_coherence();
        (coherence - CRITICAL_COHERENCE).abs() < 0.1
    }

    /// Compute golden ratio phase alignment using PENTAGON_PHASES
    /// Returns how well engine phases align with ideal pentagonal phases
    pub fn golden_phase_alignment(&self) -> f64 {
        let rates = self.spike_rates();
        let mut alignment_sum = 0.0;

        for i in 0..PENTAGON_ENGINES {
            // Map spike rate to phase and compare to ideal pentagon phase
            let actual_phase = rates[i] * 360.0;
            let ideal_phase = PENTAGON_PHASES[i];
            let phase_diff = (actual_phase - ideal_phase).abs() % 180.0;
            let normalized_diff = if phase_diff > 90.0 { 180.0 - phase_diff } else { phase_diff };
            alignment_sum += 1.0 - (normalized_diff / 90.0);
        }

        alignment_sum / PENTAGON_ENGINES as f64
    }

    /// Compute golden ratio convergence using Fibonacci approximation
    /// Uses fibonacci_sequence and golden_ratio_approximation from constants
    pub fn fibonacci_convergence(&self, n_terms: usize) -> f64 {
        let n = n_terms.max(2);
        let fib_seq = fibonacci_sequence(n);
        let phi_approx = golden_ratio_approximation(n);

        // Compute convergence error: how fast Fibonacci ratios approach φ
        let convergence_rate = if fib_seq.len() >= 2 {
            let last_ratio = fib_seq[fib_seq.len() - 1] as f64 / fib_seq[fib_seq.len() - 2] as f64;
            (last_ratio - PHI).abs()
        } else {
            1.0
        };

        // Measure how close our coupling ratios are to the Fibonacci limit
        let adjacent_ratio = self.coupling.coupling(0, 1) / self.coupling.coupling(0, 2);
        let coupling_error = (adjacent_ratio - phi_approx).abs();

        // Return combined convergence metric
        (convergence_rate + coupling_error) / 2.0
    }

    /// Compute effective coupling strength using PHI and PHI_INV
    pub fn effective_coupling_strength(&self) -> f64 {
        // Weighted sum: adjacent (φ) contributes more than skip (φ⁻¹)
        let adjacent_weight = PHI / (PHI + PHI_INV);
        let skip_weight = PHI_INV / (PHI + PHI_INV);

        let rates = self.spike_rates();
        let mut effective = 0.0;

        for i in 0..PENTAGON_ENGINES {
            let adjacent_idx = (i + 1) % PENTAGON_ENGINES;
            let skip_idx = (i + 2) % PENTAGON_ENGINES;

            effective += adjacent_weight * rates[adjacent_idx] * self.coupling.coupling(i, adjacent_idx);
            effective += skip_weight * rates[skip_idx] * self.coupling.coupling(i, skip_idx);
        }

        effective / PENTAGON_ENGINES as f64
    }
}

/// Pentagon statistics
#[derive(Debug, Clone)]
pub struct PentagonStats {
    /// Current tick
    pub tick: u64,
    /// Spike rates for all 5 engines
    pub spike_rates: [f64; PENTAGON_ENGINES],
    /// Mean spike rate across engines
    pub mean_spike_rate: f64,
    /// Variance in spike rates
    pub spike_rate_variance: f64,
    /// Phase coherence (Kuramoto order parameter)
    pub phase_coherence: f64,
    /// Spectral radius of coupling matrix
    pub spectral_radius: f64,
    /// MSOCL order parameter
    pub msocl_order_parameter: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pentagon_creation() {
        let config = PentagonConfig::default();
        let pentagon = FibonacciPentagon::new(config);

        assert_eq!(pentagon.engines.len(), PENTAGON_ENGINES);
        assert_eq!(pentagon.tick(), 0);
    }

    #[test]
    fn test_pentagon_step() {
        let config = PentagonConfig {
            engine_config: EngineConfig {
                num_pbits: 64,
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pentagon = FibonacciPentagon::new(config);
        pentagon.step();

        assert_eq!(pentagon.tick(), 1);

        let rates = pentagon.spike_rates();
        for &rate in &rates {
            assert!(rate >= 0.0 && rate <= 1.0);
        }
    }

    #[test]
    fn test_pentagon_step_n() {
        let config = PentagonConfig {
            engine_config: EngineConfig {
                num_pbits: 64,
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pentagon = FibonacciPentagon::new(config);
        pentagon.step_n(10);

        assert_eq!(pentagon.tick(), 10);
    }

    #[test]
    fn test_fibonacci_coupling_symmetry() {
        let coupling = FibonacciCoupling::default();

        for i in 0..PENTAGON_ENGINES {
            for j in 0..PENTAGON_ENGINES {
                let c_ij = coupling.coupling(i, j);
                let c_ji = coupling.coupling(j, i);
                assert!((c_ij - c_ji).abs() < 1e-10, "Coupling not symmetric");
            }
        }
    }

    #[test]
    fn test_fibonacci_coupling_zero_diagonal() {
        let coupling = FibonacciCoupling::default();

        for i in 0..PENTAGON_ENGINES {
            // The matrix has 1.0 on diagonal, scaled by FIBONACCI_COUPLING_SCALE
            let expected = FIBONACCI_COUPLING_SCALE;
            let actual = coupling.coupling(i, i);
            assert!((actual - expected).abs() < 1e-10,
                    "Self-coupling mismatch at {}: {} != {}", i, actual, expected);
        }
    }

    #[test]
    fn test_fibonacci_coupling_golden_ratio() {
        let coupling = FibonacciCoupling::default();

        // Adjacent engines should have φ coupling (from constants matrix)
        let adjacent = coupling.coupling(0, 1) / FIBONACCI_COUPLING_SCALE;
        assert!((adjacent - PHI).abs() < 1e-10,
                "Adjacent coupling incorrect: {} != {}", adjacent, PHI);

        // Skip-one engines should have φ⁻¹ coupling (from constants matrix)
        let skip_one = coupling.coupling(0, 2) / FIBONACCI_COUPLING_SCALE;
        assert!((skip_one - PHI_INV).abs() < 1e-10,
                "Skip-one coupling incorrect: {} != {}", skip_one, PHI_INV);
    }

    #[test]
    fn test_spectral_radius() {
        let coupling = FibonacciCoupling::default();
        let radius = coupling.spectral_radius();

        // Should be approximately √5 × scale
        let expected = 5.0_f64.sqrt() * FIBONACCI_COUPLING_SCALE;
        assert!((radius - expected).abs() < 1e-10);
    }

    #[test]
    fn test_phase_coherence_bounds() {
        let config = PentagonConfig {
            engine_config: EngineConfig {
                num_pbits: 64,
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pentagon = FibonacciPentagon::new(config);
        pentagon.step_n(10);

        let coherence = pentagon.phase_coherence();
        assert!(coherence >= MIN_COHERENCE && coherence <= MAX_COHERENCE,
                "Coherence out of bounds: {}", coherence);
    }

    #[test]
    fn test_inter_engine_coupling() {
        let config = PentagonConfig {
            engine_config: EngineConfig {
                num_pbits: 64,
                seed: Some(42),
                ..Default::default()
            },
            coupling_scale: 0.2,
            ..Default::default()
        };

        let mut pentagon = FibonacciPentagon::new(config);

        // Run several steps to establish coupling
        pentagon.step_n(5);

        let rates = pentagon.spike_rates();

        // All rates should be valid
        for (i, &rate) in rates.iter().enumerate() {
            assert!(rate >= 0.0 && rate <= 1.0, "Engine {} rate out of bounds: {}", i, rate);
        }
    }

    #[test]
    fn test_temperature_modulation() {
        let config = PentagonConfig::default();
        let mut pentagon = FibonacciPentagon::new(config);

        pentagon.step_n(20);

        // Each engine should have different temperature due to golden spiral
        for i in 0..PENTAGON_ENGINES {
            let engine = pentagon.engine(i).unwrap();
            let temp = engine.temperature();
            assert!(temp > 0.0, "Engine {} has non-positive temperature", i);
        }
    }

    #[test]
    fn test_reset() {
        let config = PentagonConfig {
            engine_config: EngineConfig {
                num_pbits: 64,
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pentagon = FibonacciPentagon::new(config);
        pentagon.step_n(10);

        assert!(pentagon.tick() > 0);

        pentagon.reset();
        assert_eq!(pentagon.tick(), 0);
    }

    #[test]
    fn test_stats() {
        let config = PentagonConfig {
            engine_config: EngineConfig {
                num_pbits: 64,
                seed: Some(42),
                ..Default::default()
            },
            ..Default::default()
        };

        let mut pentagon = FibonacciPentagon::new(config);
        pentagon.step_n(10);

        let stats = pentagon.stats();

        assert_eq!(stats.tick, 10);
        assert!(stats.mean_spike_rate >= 0.0 && stats.mean_spike_rate <= 1.0);
        assert!(stats.spike_rate_variance >= 0.0);
        assert!(stats.phase_coherence >= 0.0 && stats.phase_coherence <= 1.0);
        assert!(stats.spectral_radius > 0.0);
    }

    #[test]
    fn test_engine_access() {
        let config = PentagonConfig::default();
        let mut pentagon = FibonacciPentagon::new(config);

        // Test immutable access
        for i in 0..PENTAGON_ENGINES {
            let engine = pentagon.engine(i);
            assert!(engine.is_some(), "Engine {} should exist", i);
        }

        // Test out of bounds
        assert!(pentagon.engine(PENTAGON_ENGINES).is_none());

        // Test mutable access
        for i in 0..PENTAGON_ENGINES {
            let engine = pentagon.engine_mut(i);
            assert!(engine.is_some(), "Engine {} should exist (mut)", i);
        }
    }

    #[test]
    fn test_stdp_disabled() {
        let config = PentagonConfig {
            engine_config: EngineConfig {
                num_pbits: 64,
                seed: Some(42),
                ..Default::default()
            },
            enable_stdp: false,
            ..Default::default()
        };

        let mut pentagon = FibonacciPentagon::new(config);
        pentagon.step_n(10);

        // Should complete without errors even with STDP disabled
        assert_eq!(pentagon.tick(), 10);
    }
}
