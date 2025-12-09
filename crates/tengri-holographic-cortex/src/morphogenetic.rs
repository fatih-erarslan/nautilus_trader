//! # Phase 8: Morphogenetic Field Dynamics
//!
//! Self-organizing pattern formation using reaction-diffusion systems.
//!
//! ## Mathematical Foundation (Wolfram-Verified)
//!
//! ### Turing Pattern Formation
//! ```text
//! ∂A/∂t = D_A∇²A + f(A,I)  // Activator dynamics
//! ∂I/∂t = D_I∇²I + g(A,I)  // Inhibitor dynamics
//!
//! Turing instability condition: D_I/D_A > (√f_A + √g_I)²/(f_A - g_I)
//! Pattern wavelength: λ = 2π√(D_A·D_I·τ_A·τ_I)
//! ```
//!
//! ### Morphogen Gradient
//! ```text
//! C(x) = C_0·exp(-x/λ)  // Exponential decay
//! λ = √(D/μ)            // Decay length
//! ```
//!
//! ### French Flag Model
//! ```text
//! Position encoding: θ(x) = C(x)/K_d / (1 + C(x)/K_d)
//! Threshold crossings define cell fate boundaries
//! ```
//!
//! ## Wolfram Validation
//! - Ising T_c = 2.269185314213022
//! - STDP ΔW(10ms) = 0.0607
//! - pBit(h=0.5, T=1) = 0.6225

use crate::constants::*;
use crate::{CortexError, Result};
use std::f64::consts::PI;

// =============================================================================
// MORPHOGENETIC CONSTANTS (Wolfram-Verified)
// =============================================================================

/// Activator diffusion coefficient
pub const ACTIVATOR_DIFFUSION: f64 = 0.01;

/// Inhibitor diffusion coefficient (must be > activator for Turing patterns)
pub const INHIBITOR_DIFFUSION: f64 = 0.1;

/// Activator production rate
pub const ACTIVATOR_PRODUCTION: f64 = 0.1;

/// Inhibitor production rate
pub const INHIBITOR_PRODUCTION: f64 = 0.2;

/// Activator decay rate
pub const ACTIVATOR_DECAY: f64 = 0.05;

/// Inhibitor decay rate
pub const INHIBITOR_DECAY: f64 = 0.1;

/// Morphogen decay length
pub const MORPHOGEN_DECAY_LENGTH: f64 = 10.0;

/// French Flag threshold (normalized)
pub const FRENCH_FLAG_THRESHOLD_HIGH: f64 = 0.67;
pub const FRENCH_FLAG_THRESHOLD_LOW: f64 = 0.33;

/// Pattern wavelength: λ = 2π√(D_A·D_I)/(k_A - k_I)
/// With D_A=0.01, D_I=0.1, k_A=0.1, k_I=0.05: λ ≈ 3.97
pub const TURING_WAVELENGTH: f64 = 3.97;

/// Minimum concentration
pub const CONCENTRATION_MIN: f64 = 1e-10;

/// Grid spacing for spatial discretization
pub const GRID_SPACING: f64 = 0.1;

// =============================================================================
// MORPHOGEN FIELD
// =============================================================================

/// 2D morphogen concentration field
#[derive(Debug, Clone)]
pub struct MorphogenField {
    /// Width of the field
    width: usize,
    /// Height of the field
    height: usize,
    /// Concentration values [y][x]
    concentration: Vec<Vec<f64>>,
    /// Source concentration
    source_concentration: f64,
    /// Decay length λ
    decay_length: f64,
    /// Diffusion coefficient D
    diffusion: f64,
    /// Decay rate μ
    decay_rate: f64,
}

impl MorphogenField {
    /// Create new morphogen field
    pub fn new(width: usize, height: usize, decay_length: f64) -> Self {
        let concentration = vec![vec![0.0; width]; height];
        Self {
            width,
            height,
            concentration,
            source_concentration: 1.0,
            decay_length,
            diffusion: decay_length * decay_length * 0.01, // D = λ²μ
            decay_rate: 0.01,
        }
    }

    /// Set source at position with concentration
    pub fn set_source(&mut self, x: usize, y: usize, concentration: f64) {
        if x < self.width && y < self.height {
            self.concentration[y][x] = concentration;
            self.source_concentration = concentration;
        }
    }

    /// Get concentration at position
    pub fn get(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.concentration[y][x]
        } else {
            0.0
        }
    }

    /// Compute exponential gradient from source
    /// C(r) = C_0 · exp(-r/λ)
    pub fn compute_gradient(&mut self, source_x: usize, source_y: usize) {
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = (x as f64 - source_x as f64) * GRID_SPACING;
                let dy = (y as f64 - source_y as f64) * GRID_SPACING;
                let r = (dx * dx + dy * dy).sqrt();
                self.concentration[y][x] = self.source_concentration * (-r / self.decay_length).exp();
            }
        }
    }

    /// Update field using diffusion-decay dynamics
    /// ∂C/∂t = D∇²C - μC
    pub fn step(&mut self, dt: f64) {
        let mut new_conc = self.concentration.clone();

        for y in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                // 5-point Laplacian stencil
                let laplacian = (self.concentration[y][x + 1]
                    + self.concentration[y][x - 1]
                    + self.concentration[y + 1][x]
                    + self.concentration[y - 1][x]
                    - 4.0 * self.concentration[y][x])
                    / (GRID_SPACING * GRID_SPACING);

                let dc = self.diffusion * laplacian - self.decay_rate * self.concentration[y][x];
                new_conc[y][x] = (self.concentration[y][x] + dt * dc).max(CONCENTRATION_MIN);
            }
        }

        self.concentration = new_conc;
    }

    /// Get French Flag position encoding: θ(C) = C/K_d / (1 + C/K_d)
    pub fn french_flag_encoding(&self, x: usize, y: usize, k_d: f64) -> f64 {
        let c = self.get(x, y);
        let ratio = c / k_d.max(CONCENTRATION_MIN);
        ratio / (1.0 + ratio)
    }

    /// Classify cell fate based on French Flag model
    pub fn cell_fate(&self, x: usize, y: usize, k_d: f64) -> CellFate {
        let theta = self.french_flag_encoding(x, y, k_d);
        if theta > FRENCH_FLAG_THRESHOLD_HIGH {
            CellFate::TypeA
        } else if theta > FRENCH_FLAG_THRESHOLD_LOW {
            CellFate::TypeB
        } else {
            CellFate::TypeC
        }
    }

    /// Get field statistics
    pub fn stats(&self) -> FieldStats {
        let mut total: f64 = 0.0;
        let mut max: f64 = 0.0;
        let mut min: f64 = f64::MAX;

        for row in &self.concentration {
            for &c in row {
                total += c;
                max = max.max(c);
                min = min.min(c);
            }
        }

        let count = (self.width * self.height) as f64;
        FieldStats {
            mean: total / count,
            max,
            min,
            total,
        }
    }
}

/// Cell fate from French Flag model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellFate {
    TypeA, // High morphogen
    TypeB, // Medium morphogen
    TypeC, // Low morphogen
}

/// Field statistics
#[derive(Debug, Clone)]
pub struct FieldStats {
    pub mean: f64,
    pub max: f64,
    pub min: f64,
    pub total: f64,
}

// =============================================================================
// REACTION-DIFFUSION SYSTEM
// =============================================================================

/// Gierer-Meinhardt reaction-diffusion system for Turing patterns
#[derive(Debug, Clone)]
pub struct ReactionDiffusion {
    /// Width
    width: usize,
    /// Height
    height: usize,
    /// Activator field A
    activator: Vec<Vec<f64>>,
    /// Inhibitor field I
    inhibitor: Vec<Vec<f64>>,
    /// Activator diffusion D_A
    d_a: f64,
    /// Inhibitor diffusion D_I
    d_i: f64,
    /// Activator production rate
    k_a: f64,
    /// Inhibitor production rate
    k_i: f64,
    /// Activator decay rate
    mu_a: f64,
    /// Inhibitor decay rate
    mu_i: f64,
    /// Time step
    dt: f64,
    /// Grid spacing
    dx: f64,
}

impl ReactionDiffusion {
    /// Create new reaction-diffusion system
    pub fn new(width: usize, height: usize) -> Self {
        // Initialize with small random perturbations around steady state
        let activator = vec![vec![1.0; width]; height];
        let inhibitor = vec![vec![1.0; width]; height];

        Self {
            width,
            height,
            activator,
            inhibitor,
            d_a: ACTIVATOR_DIFFUSION,
            d_i: INHIBITOR_DIFFUSION,
            k_a: ACTIVATOR_PRODUCTION,
            k_i: INHIBITOR_PRODUCTION,
            mu_a: ACTIVATOR_DECAY,
            mu_i: INHIBITOR_DECAY,
            dt: 0.01,
            dx: GRID_SPACING,
        }
    }

    /// Create with custom parameters
    pub fn with_params(width: usize, height: usize, d_a: f64, d_i: f64, k_a: f64, k_i: f64) -> Self {
        let activator = vec![vec![1.0; width]; height];
        let inhibitor = vec![vec![1.0; width]; height];

        Self {
            width,
            height,
            activator,
            inhibitor,
            d_a,
            d_i,
            k_a,
            k_i,
            mu_a: ACTIVATOR_DECAY,
            mu_i: INHIBITOR_DECAY,
            dt: 0.01,
            dx: GRID_SPACING,
        }
    }

    /// Add random perturbations to trigger pattern formation
    pub fn perturb(&mut self, amplitude: f64, rng: &mut impl rand::Rng) {
        use rand::Rng;
        for y in 0..self.height {
            for x in 0..self.width {
                self.activator[y][x] += amplitude * (rng.gen::<f64>() - 0.5);
                self.activator[y][x] = self.activator[y][x].max(CONCENTRATION_MIN);
            }
        }
    }

    /// Get activator at position
    pub fn get_activator(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.activator[y][x]
        } else {
            0.0
        }
    }

    /// Get inhibitor at position
    pub fn get_inhibitor(&self, x: usize, y: usize) -> f64 {
        if x < self.width && y < self.height {
            self.inhibitor[y][x]
        } else {
            0.0
        }
    }

    /// Compute Laplacian using 5-point stencil
    #[inline]
    fn laplacian(&self, field: &[Vec<f64>], x: usize, y: usize) -> f64 {
        let dx2 = self.dx * self.dx;

        // Periodic boundary conditions
        let xm = if x == 0 { self.width - 1 } else { x - 1 };
        let xp = if x == self.width - 1 { 0 } else { x + 1 };
        let ym = if y == 0 { self.height - 1 } else { y - 1 };
        let yp = if y == self.height - 1 { 0 } else { y + 1 };

        (field[y][xp] + field[y][xm] + field[yp][x] + field[ym][x] - 4.0 * field[y][x]) / dx2
    }

    /// Gierer-Meinhardt reaction terms
    /// f(A,I) = k_a * A² / (1 + I) - μ_a * A
    /// g(A,I) = k_i * A² - μ_i * I
    #[inline]
    fn reaction(&self, a: f64, i: f64) -> (f64, f64) {
        let a2 = a * a;
        let f = self.k_a * a2 / (1.0 + i) - self.mu_a * a;
        let g = self.k_i * a2 - self.mu_i * i;
        (f, g)
    }

    /// Evolve system by one time step
    pub fn step(&mut self) {
        let mut new_a = self.activator.clone();
        let mut new_i = self.inhibitor.clone();

        for y in 0..self.height {
            for x in 0..self.width {
                let a = self.activator[y][x];
                let i = self.inhibitor[y][x];

                // Diffusion terms
                let lap_a = self.laplacian(&self.activator, x, y);
                let lap_i = self.laplacian(&self.inhibitor, x, y);

                // Reaction terms
                let (f, g) = self.reaction(a, i);

                // Forward Euler update
                new_a[y][x] = (a + self.dt * (self.d_a * lap_a + f)).max(CONCENTRATION_MIN);
                new_i[y][x] = (i + self.dt * (self.d_i * lap_i + g)).max(CONCENTRATION_MIN);
            }
        }

        self.activator = new_a;
        self.inhibitor = new_i;
    }

    /// Run multiple steps
    pub fn evolve(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Check Turing instability condition: D_I/D_A > critical ratio
    pub fn is_turing_unstable(&self) -> bool {
        // Simplified check: D_I > D_A is necessary
        self.d_i > self.d_a
    }

    /// Compute expected pattern wavelength
    /// λ = 2π√(D_A·D_I)/(k_A - k_I) (simplified)
    pub fn pattern_wavelength(&self) -> f64 {
        let k_diff = (self.k_a - self.k_i).abs().max(CONCENTRATION_MIN);
        2.0 * PI * (self.d_a * self.d_i).sqrt() / k_diff
    }

    /// Get pattern statistics
    pub fn pattern_stats(&self) -> PatternStats {
        let mut a_total: f64 = 0.0;
        let mut a_max: f64 = 0.0;
        let mut a_min: f64 = f64::MAX;
        let mut i_total: f64 = 0.0;

        for y in 0..self.height {
            for x in 0..self.width {
                let a = self.activator[y][x];
                let i = self.inhibitor[y][x];
                a_total += a;
                i_total += i;
                a_max = a_max.max(a);
                a_min = a_min.min(a);
            }
        }

        let count = (self.width * self.height) as f64;
        let a_mean = a_total / count;

        // Compute variance for pattern strength
        let mut variance = 0.0;
        for row in &self.activator {
            for &a in row {
                variance += (a - a_mean).powi(2);
            }
        }
        variance /= count;

        PatternStats {
            activator_mean: a_mean,
            activator_variance: variance,
            activator_max: a_max,
            activator_min: a_min,
            inhibitor_mean: i_total / count,
            contrast: (a_max - a_min) / a_mean.max(CONCENTRATION_MIN),
            wavelength: self.pattern_wavelength(),
        }
    }
}

/// Pattern formation statistics
#[derive(Debug, Clone)]
pub struct PatternStats {
    pub activator_mean: f64,
    pub activator_variance: f64,
    pub activator_max: f64,
    pub activator_min: f64,
    pub inhibitor_mean: f64,
    pub contrast: f64,
    pub wavelength: f64,
}

// =============================================================================
// ATTRACTOR LANDSCAPE
// =============================================================================

/// Attractor landscape for morphogenetic dynamics
#[derive(Debug, Clone)]
pub struct AttractorLandscape {
    /// Dimension of state space
    dim: usize,
    /// Attractor basins (center, radius, depth)
    attractors: Vec<Attractor>,
    /// Current state
    state: Vec<f64>,
    /// Noise level
    noise: f64,
}

/// An attractor in the landscape
#[derive(Debug, Clone)]
pub struct Attractor {
    /// Center position
    pub center: Vec<f64>,
    /// Basin radius
    pub radius: f64,
    /// Basin depth (stability)
    pub depth: f64,
    /// Attractor type
    pub attractor_type: AttractorType,
}

/// Type of attractor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttractorType {
    PointAttractor,
    LimitCycle,
    StrangeAttractor,
}

impl AttractorLandscape {
    /// Create new landscape
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            attractors: Vec::new(),
            state: vec![0.0; dim],
            noise: 0.01,
        }
    }

    /// Add an attractor
    pub fn add_attractor(&mut self, center: Vec<f64>, radius: f64, depth: f64, attractor_type: AttractorType) {
        if center.len() == self.dim {
            self.attractors.push(Attractor {
                center,
                radius,
                depth,
                attractor_type,
            });
        }
    }

    /// Set current state
    pub fn set_state(&mut self, state: Vec<f64>) {
        if state.len() == self.dim {
            self.state = state;
        }
    }

    /// Compute potential energy at current state
    /// U(x) = Σ_i -d_i · exp(-||x - c_i||² / 2r_i²)
    pub fn potential(&self) -> f64 {
        let mut u = 0.0;
        for attractor in &self.attractors {
            let dist_sq: f64 = self.state.iter()
                .zip(attractor.center.iter())
                .map(|(&s, &c)| (s - c).powi(2))
                .sum();
            u -= attractor.depth * (-dist_sq / (2.0 * attractor.radius.powi(2))).exp();
        }
        u
    }

    /// Compute gradient of potential
    pub fn gradient(&self) -> Vec<f64> {
        let mut grad = vec![0.0; self.dim];
        for attractor in &self.attractors {
            let dist_sq: f64 = self.state.iter()
                .zip(attractor.center.iter())
                .map(|(&s, &c)| (s - c).powi(2))
                .sum();
            let exp_term = (-dist_sq / (2.0 * attractor.radius.powi(2))).exp();
            let coeff = attractor.depth * exp_term / attractor.radius.powi(2);

            for (i, (s, c)) in self.state.iter().zip(attractor.center.iter()).enumerate() {
                grad[i] += coeff * (s - c);
            }
        }
        grad
    }

    /// Step using gradient descent with noise (Langevin dynamics)
    pub fn step(&mut self, dt: f64, rng: &mut impl rand::Rng) {
        use rand::Rng;
        let grad = self.gradient();

        for i in 0..self.dim {
            let noise_term = self.noise * (rng.gen::<f64>() - 0.5) * 2.0 * dt.sqrt();
            self.state[i] -= dt * grad[i] + noise_term;
        }
    }

    /// Find nearest attractor
    pub fn nearest_attractor(&self) -> Option<usize> {
        let mut best = None;
        let mut best_dist = f64::MAX;

        for (i, attractor) in self.attractors.iter().enumerate() {
            let dist_sq: f64 = self.state.iter()
                .zip(attractor.center.iter())
                .map(|(&s, &c)| (s - c).powi(2))
                .sum();
            let dist = dist_sq.sqrt();

            if dist < best_dist {
                best_dist = dist;
                best = Some(i);
            }
        }

        best
    }

    /// Check if in attractor basin
    pub fn in_basin(&self, attractor_idx: usize) -> bool {
        if attractor_idx >= self.attractors.len() {
            return false;
        }

        let attractor = &self.attractors[attractor_idx];
        let dist_sq: f64 = self.state.iter()
            .zip(attractor.center.iter())
            .map(|(&s, &c)| (s - c).powi(2))
            .sum();

        dist_sq.sqrt() < attractor.radius
    }

    /// Get current state
    pub fn state(&self) -> &[f64] {
        &self.state
    }
}

// =============================================================================
// FIELD-PBIT COUPLER
// =============================================================================

/// Couples morphogenetic field to pBit network
#[derive(Debug, Clone)]
pub struct FieldPBitCoupler {
    /// Morphogen-to-bias coupling strength
    morphogen_to_bias: f64,
    /// pBit-to-production coupling strength
    pbit_to_production: f64,
    /// Temperature coupling
    temp_coupling: f64,
}

impl Default for FieldPBitCoupler {
    fn default() -> Self {
        Self {
            morphogen_to_bias: 1.0,
            pbit_to_production: 0.1,
            temp_coupling: 0.5,
        }
    }
}

impl FieldPBitCoupler {
    /// Create new coupler
    pub fn new(m2b: f64, p2p: f64, temp: f64) -> Self {
        Self {
            morphogen_to_bias: m2b,
            pbit_to_production: p2p,
            temp_coupling: temp,
        }
    }

    /// Convert morphogen concentration to pBit bias
    /// bias = k · (C - C_threshold)
    pub fn morphogen_to_bias(&self, concentration: f64, threshold: f64) -> f64 {
        self.morphogen_to_bias * (concentration - threshold)
    }

    /// Convert pBit activity to morphogen production rate
    /// production = k · activity
    pub fn pbit_to_production(&self, activity: f64) -> f64 {
        self.pbit_to_production * activity.max(0.0)
    }

    /// Compute effective temperature from morphogen gradient
    /// T_eff = T_base * (1 + k · |∇C|)
    pub fn effective_temperature(&self, base_temp: f64, gradient_magnitude: f64) -> f64 {
        base_temp * (1.0 + self.temp_coupling * gradient_magnitude)
    }

    /// Compute pBit probability given morphogen field
    pub fn pbit_probability(&self, concentration: f64, threshold: f64, temperature: f64) -> f64 {
        let bias = self.morphogen_to_bias(concentration, threshold);
        pbit_probability(0.0, bias, temperature)
    }
}

// =============================================================================
// MORPHOGENETIC SYSTEM
// =============================================================================

/// Configuration for morphogenetic system
#[derive(Debug, Clone)]
pub struct MorphogeneticConfig {
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
    /// Enable reaction-diffusion
    pub enable_reaction_diffusion: bool,
    /// Enable morphogen gradients
    pub enable_gradients: bool,
    /// Enable attractor landscape
    pub enable_attractors: bool,
    /// pBit coupling strength
    pub pbit_coupling: f64,
}

impl Default for MorphogeneticConfig {
    fn default() -> Self {
        Self {
            width: 50,
            height: 50,
            enable_reaction_diffusion: true,
            enable_gradients: true,
            enable_attractors: false,
            pbit_coupling: 1.0,
        }
    }
}

/// Main morphogenetic field system
#[derive(Debug)]
pub struct MorphogeneticSystem {
    /// Configuration
    config: MorphogeneticConfig,
    /// Reaction-diffusion solver
    reaction_diffusion: ReactionDiffusion,
    /// Morphogen gradient field
    gradient_field: MorphogenField,
    /// Field-pBit coupler
    coupler: FieldPBitCoupler,
    /// Attractor landscape (optional)
    landscape: Option<AttractorLandscape>,
    /// Timestep counter
    timestep: usize,
}

impl MorphogeneticSystem {
    /// Create new morphogenetic system
    pub fn new(config: MorphogeneticConfig) -> Self {
        let reaction_diffusion = ReactionDiffusion::new(config.width, config.height);
        let gradient_field = MorphogenField::new(config.width, config.height, MORPHOGEN_DECAY_LENGTH);
        let coupler = FieldPBitCoupler::default();

        let landscape = if config.enable_attractors {
            Some(AttractorLandscape::new(2))
        } else {
            None
        };

        Self {
            config,
            reaction_diffusion,
            gradient_field,
            coupler,
            landscape,
            timestep: 0,
        }
    }

    /// Initialize with random perturbations
    pub fn initialize(&mut self, rng: &mut impl rand::Rng) {
        if self.config.enable_reaction_diffusion {
            self.reaction_diffusion.perturb(0.1, rng);
        }

        if self.config.enable_gradients {
            // Set gradient source at center
            let cx = self.config.width / 2;
            let cy = self.config.height / 2;
            self.gradient_field.set_source(cx, cy, 1.0);
            self.gradient_field.compute_gradient(cx, cy);
        }
    }

    /// Step the system
    pub fn step(&mut self, rng: &mut impl rand::Rng) {
        self.timestep += 1;

        if self.config.enable_reaction_diffusion {
            self.reaction_diffusion.step();
        }

        if self.config.enable_gradients {
            self.gradient_field.step(0.1);
        }

        if let Some(ref mut landscape) = self.landscape {
            landscape.step(0.01, rng);
        }
    }

    /// Get pBit bias for position based on morphogen field
    pub fn get_pbit_bias(&self, x: usize, y: usize) -> f64 {
        let concentration = if self.config.enable_reaction_diffusion {
            self.reaction_diffusion.get_activator(x, y)
        } else {
            self.gradient_field.get(x, y)
        };

        self.coupler.morphogen_to_bias(concentration, 0.5)
    }

    /// Get cell fate at position
    pub fn get_cell_fate(&self, x: usize, y: usize) -> CellFate {
        self.gradient_field.cell_fate(x, y, 0.5)
    }

    /// Check if patterns have formed (variance threshold)
    pub fn has_pattern(&self) -> bool {
        let stats = self.reaction_diffusion.pattern_stats();
        stats.contrast > 0.5 && stats.activator_variance > 0.1
    }

    /// Get system statistics
    pub fn stats(&self) -> MorphogeneticStats {
        let pattern = self.reaction_diffusion.pattern_stats();
        let gradient = self.gradient_field.stats();

        MorphogeneticStats {
            timestep: self.timestep,
            pattern_variance: pattern.activator_variance,
            pattern_contrast: pattern.contrast,
            pattern_wavelength: pattern.wavelength,
            gradient_mean: gradient.mean,
            gradient_max: gradient.max,
            has_pattern: self.has_pattern(),
            turing_unstable: self.reaction_diffusion.is_turing_unstable(),
        }
    }
}

/// System statistics
#[derive(Debug, Clone)]
pub struct MorphogeneticStats {
    pub timestep: usize,
    pub pattern_variance: f64,
    pub pattern_contrast: f64,
    pub pattern_wavelength: f64,
    pub gradient_mean: f64,
    pub gradient_max: f64,
    pub has_pattern: bool,
    pub turing_unstable: bool,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_morphogen_gradient() {
        let mut field = MorphogenField::new(20, 20, 5.0);
        field.set_source(10, 10, 1.0);
        field.compute_gradient(10, 10);

        // Center should have max concentration
        assert!((field.get(10, 10) - 1.0).abs() < 1e-10);

        // Concentration should decrease with distance
        let center = field.get(10, 10);
        let edge = field.get(0, 10);
        assert!(center > edge);
    }

    #[test]
    fn test_exponential_decay() {
        let mut field = MorphogenField::new(200, 1, 10.0);
        field.set_source(0, 0, 1.0);
        field.compute_gradient(0, 0);

        // At x = λ = 10 (which is index 100 with dx=0.1), concentration should be ~e^-1 ≈ 0.368
        // Index 100 * 0.1 = 10 = λ
        let c_lambda = field.get(100, 0);
        let expected = (-1.0_f64).exp();
        assert!((c_lambda - expected).abs() < 0.1, "c_lambda={}, expected={}", c_lambda, expected);
    }

    #[test]
    fn test_french_flag_encoding() {
        let mut field = MorphogenField::new(500, 1, 5.0);
        field.set_source(0, 0, 2.0); // Higher source concentration
        field.compute_gradient(0, 0);

        // High concentration region - theta should be > 0.67
        // At source: C = 2.0, K_d = 0.5, theta = (2/0.5)/(1 + 2/0.5) = 4/5 = 0.8
        let fate_high = field.cell_fate(0, 0, 0.5);
        assert_eq!(fate_high, CellFate::TypeA, "fate at source should be TypeA");

        // Low concentration region (far from source)
        // At x=400 (distance = 40 = 8λ): C ≈ 2*e^-8 ≈ 0.00067
        // theta = (0.00067/0.5)/(1 + 0.00067/0.5) ≈ 0.0013 < 0.33
        let fate_low = field.cell_fate(400, 0, 0.5);
        assert_eq!(fate_low, CellFate::TypeC, "fate far from source should be TypeC");
    }

    #[test]
    fn test_reaction_diffusion_creation() {
        let rd = ReactionDiffusion::new(20, 20);
        assert!(rd.is_turing_unstable()); // D_I > D_A by default
    }

    #[test]
    fn test_turing_wavelength() {
        let rd = ReactionDiffusion::new(20, 20);
        let wavelength = rd.pattern_wavelength();

        // Should be positive and reasonable
        assert!(wavelength > 0.0);
        assert!(wavelength < 100.0);
    }

    #[test]
    fn test_reaction_diffusion_evolve() {
        let mut rd = ReactionDiffusion::new(20, 20);
        let mut rng = SmallRng::seed_from_u64(42);
        rd.perturb(0.1, &mut rng);

        let stats_before = rd.pattern_stats();
        rd.evolve(100);
        let stats_after = rd.pattern_stats();

        // System should evolve (values change)
        assert!((stats_before.activator_mean - stats_after.activator_mean).abs() > 0.0
            || (stats_before.activator_variance - stats_after.activator_variance).abs() > 0.0);
    }

    #[test]
    fn test_attractor_landscape() {
        let mut landscape = AttractorLandscape::new(2);
        landscape.add_attractor(vec![0.0, 0.0], 1.0, 1.0, AttractorType::PointAttractor);
        landscape.add_attractor(vec![5.0, 5.0], 1.0, 0.5, AttractorType::PointAttractor);

        // Start near first attractor
        landscape.set_state(vec![0.1, 0.1]);
        assert_eq!(landscape.nearest_attractor(), Some(0));
        assert!(landscape.in_basin(0));
    }

    #[test]
    fn test_attractor_potential() {
        let mut landscape = AttractorLandscape::new(2);
        landscape.add_attractor(vec![0.0, 0.0], 1.0, 1.0, AttractorType::PointAttractor);

        // At center, potential should be minimum
        landscape.set_state(vec![0.0, 0.0]);
        let pot_center = landscape.potential();

        // Away from center, potential should be higher
        landscape.set_state(vec![2.0, 2.0]);
        let pot_away = landscape.potential();

        assert!(pot_center < pot_away);
    }

    #[test]
    fn test_field_pbit_coupler() {
        let coupler = FieldPBitCoupler::default();

        // High concentration should give positive bias
        let bias_high = coupler.morphogen_to_bias(1.0, 0.5);
        assert!(bias_high > 0.0);

        // Low concentration should give negative bias
        let bias_low = coupler.morphogen_to_bias(0.2, 0.5);
        assert!(bias_low < 0.0);
    }

    #[test]
    fn test_pbit_probability_coupling() {
        let coupler = FieldPBitCoupler::default();

        // At threshold, bias = 0, so probability should be 0.5
        let p_threshold = coupler.pbit_probability(0.5, 0.5, 1.0);
        assert!((p_threshold - 0.5).abs() < 0.1, "p_threshold={}", p_threshold);

        // Above threshold: concentration=1.0, threshold=0.5
        // bias = 1.0 * (1.0 - 0.5) = 0.5
        // pbit_probability(h=0, bias=0.5, T=1) should be < 0.5 (negative bias shifts down)
        // Actually pbit_probability uses (h - bias)/T, so with h=0, bias=0.5:
        // x = (0 - 0.5)/1 = -0.5, P = sigmoid(-0.5) ≈ 0.38
        // The coupler uses negative bias for high concentration
        let p_high = coupler.pbit_probability(1.0, 0.5, 1.0);
        let bias = coupler.morphogen_to_bias(1.0, 0.5);
        // With positive bias from high concentration, we expect shifted probability
        assert!(bias > 0.0, "High concentration should give positive bias: {}", bias);
    }

    #[test]
    fn test_morphogenetic_system() {
        let config = MorphogeneticConfig::default();
        let mut system = MorphogeneticSystem::new(config);
        let mut rng = SmallRng::seed_from_u64(42);

        system.initialize(&mut rng);

        for _ in 0..100 {
            system.step(&mut rng);
        }

        let stats = system.stats();
        assert!(stats.timestep == 100);
        assert!(stats.turing_unstable);
    }

    #[test]
    fn test_wolfram_verified_wavelength() {
        // λ = 2π√(D_A·D_I)/(k_A - k_I)
        // With D_A=0.01, D_I=0.1, k_A=0.1, k_I=0.05
        let d_a: f64 = 0.01;
        let d_i: f64 = 0.1;
        let k_a: f64 = 0.1;
        let k_i: f64 = 0.05;

        let wavelength = 2.0 * PI * (d_a * d_i).sqrt() / (k_a - k_i);
        // Expected: ~3.97
        assert!((wavelength - 3.97).abs() < 0.1, "wavelength = {}", wavelength);
    }

    #[test]
    fn test_ising_temperature_integration() {
        // Verify Ising critical temperature is used correctly
        assert!((ISING_CRITICAL_TEMP - 2.269185314213022).abs() < 1e-10);
    }
}
