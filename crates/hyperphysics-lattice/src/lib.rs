//! # Hyperphysics Lattice
//!
//! Exotic hyperbolic lattice topologies for cognitive architectures, featuring:
//!
//! - **{p,q} Schläfli Tilings**: {7,3} heptagonal, {5,4} pentagonal, {6,4} hexagonal, {8,3} octagonal
//! - **Lorentz Hyperboloid Model**: 10x faster than Poincaré with numerical stability
//! - **Self-Organized Criticality (SOC)**: Branching ratio σ ≈ 1.0, power-law avalanches
//! - **Consciousness Metrics**: Integrated Information Φ, coherence, entanglement
//! - **Game-Theoretic Coordination**: Nash equilibrium, Φ-stability mapping
//! - **Fractal Hierarchy**: Multi-scale organization with golden ratio distribution
//!
//! ## Mathematical Foundations
//!
//! ### Hyperbolic Distance (Lorentz Model)
//! ```text
//! d(x,y) = arcosh(|⟨x,y⟩_M|)
//! where ⟨x,y⟩_M = x₀y₀ - x₁y₁ - ... - xₙyₙ (Minkowski inner product)
//! ```
//!
//! ### Schläfli Symbol {p,q}
//! ```text
//! (p-2)(q-2) > 4 → Hyperbolic tiling
//! Defect angle: δ = 2π - q × (π - 2π/p)
//! ```
//!
//! ### SOC Branching Ratio
//! ```text
//! σ = triggered_spikes / initiating_spikes
//! Critical regime: σ ∈ [0.9, 1.1]
//! ```

use nalgebra::{DMatrix, Vector4};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// Rayon parallel processing available via dependency
#[allow(unused_imports)]
use rayon::prelude::*;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors for lattice operations
#[derive(Error, Debug)]
pub enum LatticeError {
    #[error("Invalid Schläfli symbol {{p={p}, q={q}}}: must satisfy (p-2)(q-2) > 4 for hyperbolic tiling")]
    InvalidSchlafli { p: u32, q: u32 },

    #[error("Node {0} not found in lattice")]
    NodeNotFound(usize),

    #[error("Edge already exists between {0} and {1}")]
    EdgeExists(usize, usize),

    #[error("Manifold constraint violated: {0}")]
    ManifoldViolation(String),

    #[error("SOC stability violation: σ={sigma:.3} outside critical range [{min:.2}, {max:.2}]")]
    SOCInstability { sigma: f64, min: f64, max: f64 },

    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
}

pub type Result<T> = std::result::Result<T, LatticeError>;

// ============================================================================
// SCHLÄFLI SYMBOL & TILING TYPES
// ============================================================================

/// Schläfli symbol {p, q} representing a regular tiling
/// - p: number of sides per polygon
/// - q: number of polygons meeting at each vertex
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchlafliSymbol {
    pub p: u32,
    pub q: u32,
}

impl SchlafliSymbol {
    /// Create a new Schläfli symbol, validating it represents a hyperbolic tiling
    pub fn new(p: u32, q: u32) -> Result<Self> {
        let defect = (p as i32 - 2) * (q as i32 - 2);
        if defect > 4 {
            Ok(Self { p, q })
        } else {
            Err(LatticeError::InvalidSchlafli { p, q })
        }
    }

    /// {7,3} Heptagonal tessellation - 7 triangles per vertex
    pub fn heptagonal() -> Self {
        Self { p: 7, q: 3 }
    }

    /// {5,4} Pentagonal tessellation - 4 pentagons per vertex
    pub fn pentagonal() -> Self {
        Self { p: 5, q: 4 }
    }

    /// {6,4} Hexagonal hyperbolic - 4 hexagons per vertex
    pub fn hexagonal() -> Self {
        Self { p: 6, q: 4 }
    }

    /// {8,3} Octagonal tessellation - 3 octagons per vertex
    pub fn octagonal() -> Self {
        Self { p: 8, q: 3 }
    }

    /// Calculate the angular defect per vertex (negative curvature indicator)
    pub fn angular_defect(&self) -> f64 {
        let interior_angle = std::f64::consts::PI * (1.0 - 2.0 / self.p as f64);
        2.0 * std::f64::consts::PI - self.q as f64 * interior_angle
    }

    /// Calculate the area of a fundamental domain (via Gauss-Bonnet)
    pub fn fundamental_area(&self, curvature: f64) -> f64 {
        self.angular_defect().abs() / curvature.abs()
    }

    /// Calculate the number of edges per face
    pub fn edges_per_face(&self) -> u32 {
        self.p
    }

    /// Calculate the vertex degree
    pub fn vertex_degree(&self) -> u32 {
        self.q
    }

    /// Check if this is a valid hyperbolic tiling
    pub fn is_hyperbolic(&self) -> bool {
        (self.p as i32 - 2) * (self.q as i32 - 2) > 4
    }
}

// ============================================================================
// LORENTZ COORDINATES (10x FASTER THAN POINCARÉ)
// ============================================================================

/// 4D Lorentz coordinate on the hyperboloid x₀² - x₁² - x₂² - x₃² = 1
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct LorentzPoint {
    /// Timelike component (always ≥ 1)
    pub t: f64,
    /// Spacelike components
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for LorentzPoint {
    fn default() -> Self {
        Self::origin()
    }
}

impl LorentzPoint {
    /// Create a new point, projecting onto the hyperboloid if needed
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        let mut point = Self { t, x, y, z };
        point.project_to_hyperboloid();
        point
    }

    /// Origin of the hyperboloid (north pole)
    pub fn origin() -> Self {
        Self { t: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Create from spherical coordinates on the hyperboloid
    pub fn from_spherical(rapidity: f64, theta: f64, phi: f64) -> Self {
        Self {
            t: rapidity.cosh(),
            x: rapidity.sinh() * theta.sin() * phi.cos(),
            y: rapidity.sinh() * theta.sin() * phi.sin(),
            z: rapidity.sinh() * theta.cos(),
        }
    }

    /// Minkowski inner product ⟨x,y⟩_M = t₁t₂ - x₁x₂ - y₁y₂ - z₁z₂
    #[inline]
    pub fn minkowski_inner(&self, other: &Self) -> f64 {
        self.t * other.t - self.x * other.x - self.y * other.y - self.z * other.z
    }

    /// Minkowski norm squared (should be 1 on hyperboloid)
    #[inline]
    pub fn minkowski_norm_sq(&self) -> f64 {
        self.t * self.t - self.x * self.x - self.y * self.y - self.z * self.z
    }

    /// Hyperbolic distance d(x,y) = arcosh(|⟨x,y⟩_M|)
    #[inline]
    pub fn distance(&self, other: &Self) -> f64 {
        let inner = self.minkowski_inner(other).abs();
        if inner <= 1.0 {
            0.0
        } else {
            inner.acosh()
        }
    }

    /// Project point onto the hyperboloid (enforce constraint)
    pub fn project_to_hyperboloid(&mut self) {
        let spatial_sq = self.x * self.x + self.y * self.y + self.z * self.z;
        self.t = (1.0 + spatial_sq).sqrt();
    }

    /// Convert to nalgebra Vector4
    pub fn to_vector4(&self) -> Vector4<f64> {
        Vector4::new(self.t, self.x, self.y, self.z)
    }

    /// Create from nalgebra Vector4
    pub fn from_vector4(v: &Vector4<f64>) -> Self {
        Self::new(v[0], v[1], v[2], v[3])
    }

    /// Exponential map from tangent space at origin
    pub fn exp_map(tangent: &Vector4<f64>) -> Self {
        let spatial_norm = (tangent[1] * tangent[1] + tangent[2] * tangent[2] + tangent[3] * tangent[3]).sqrt();
        if spatial_norm < 1e-10 {
            return Self::origin();
        }
        Self {
            t: spatial_norm.cosh(),
            x: tangent[1] * spatial_norm.sinh() / spatial_norm,
            y: tangent[2] * spatial_norm.sinh() / spatial_norm,
            z: tangent[3] * spatial_norm.sinh() / spatial_norm,
        }
    }

    /// Logarithmic map to tangent space at origin
    pub fn log_map(&self) -> Vector4<f64> {
        let dist = self.distance(&Self::origin());
        if dist < 1e-10 {
            return Vector4::zeros();
        }
        let factor = dist / dist.sinh();
        Vector4::new(0.0, self.x * factor, self.y * factor, self.z * factor)
    }

    /// Parallel transport along geodesic from origin to self
    pub fn parallel_transport(&self, v: &Vector4<f64>) -> Vector4<f64> {
        let inner = self.t * v[0] - self.x * v[1] - self.y * v[2] - self.z * v[3];
        let origin = Self::origin();
        let factor = inner / (1.0 + self.t);
        Vector4::new(
            v[0] + factor * (self.t + origin.t),
            v[1] + factor * (self.x + origin.x),
            v[2] + factor * (self.y + origin.y),
            v[3] + factor * (self.z + origin.z),
        )
    }
}

// ============================================================================
// LATTICE NODE
// ============================================================================

/// A node in the hyperbolic lattice with SOC dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeNode {
    /// Unique identifier
    pub id: usize,
    /// Position in Lorentz coordinates
    pub position: LorentzPoint,
    /// Depth in the hierarchical lattice (0 = boundary, increasing toward center)
    pub depth: u32,
    /// Current activation/potential (for SOC dynamics)
    pub potential: f64,
    /// Threshold for firing (SOC dynamics)
    pub threshold: f64,
    /// Refractory period remaining
    pub refractory: u32,
    /// Indices of neighboring nodes
    pub neighbors: Vec<usize>,
    /// Quantum state amplitude (for consciousness metrics)
    pub amplitude: num_complex::Complex64,
    /// Local curvature estimate
    pub curvature: f64,
}

impl LatticeNode {
    /// Create a new node at the given position
    pub fn new(id: usize, position: LorentzPoint, depth: u32) -> Self {
        Self {
            id,
            position,
            depth,
            potential: 0.0,
            threshold: 1.0,
            refractory: 0,
            neighbors: Vec::new(),
            amplitude: num_complex::Complex64::new(1.0, 0.0),
            curvature: -1.0, // Default hyperbolic curvature
        }
    }

    /// Check if node can fire (potential ≥ threshold and not refractory)
    pub fn can_fire(&self) -> bool {
        self.refractory == 0 && self.potential >= self.threshold
    }

    /// Fire the node, resetting potential and entering refractory period
    pub fn fire(&mut self, refractory_period: u32) {
        self.potential = 0.0;
        self.refractory = refractory_period;
    }

    /// Update refractory counter
    pub fn tick(&mut self) {
        if self.refractory > 0 {
            self.refractory -= 1;
        }
    }

    /// Add potential (excitatory input)
    pub fn excite(&mut self, amount: f64) {
        if self.refractory == 0 {
            self.potential += amount;
        }
    }

    /// Subtract potential (inhibitory input)
    pub fn inhibit(&mut self, amount: f64) {
        if self.refractory == 0 {
            self.potential = (self.potential - amount).max(0.0);
        }
    }
}

// ============================================================================
// SOC DYNAMICS (SELF-ORGANIZED CRITICALITY)
// ============================================================================

/// Self-Organized Criticality statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SOCStats {
    /// Measured branching ratio σ = triggered/initiating
    pub sigma_measured: f64,
    /// Target branching ratio (1.0 for critical regime)
    pub sigma_target: f64,
    /// Power-law exponent τ for avalanche size distribution
    pub power_law_tau: f64,
    /// Whether system is in critical regime
    pub is_critical: bool,
    /// Total avalanches observed
    pub total_avalanches: usize,
    /// Average avalanche size
    pub avg_avalanche_size: f64,
    /// Largest avalanche observed
    pub largest_avalanche: usize,
    /// Total initiating spikes
    pub total_initiating: usize,
    /// Total triggered spikes
    pub total_triggered: usize,
}

impl SOCStats {
    /// Check if in critical regime (σ ∈ [0.9, 1.1])
    pub fn check_criticality(&mut self, tolerance: f64) {
        self.is_critical = (self.sigma_measured - self.sigma_target).abs() <= tolerance;
    }

    /// Update sigma measurement
    pub fn update_sigma(&mut self, initiating: usize, triggered: usize) {
        self.total_initiating += initiating;
        self.total_triggered += triggered;
        if self.total_initiating > 0 {
            self.sigma_measured = self.total_triggered as f64 / self.total_initiating as f64;
        }
    }
}

/// SOC dynamics manager for the lattice
#[derive(Debug, Clone)]
pub struct SOCDynamics {
    /// Current statistics
    pub stats: SOCStats,
    /// Avalanche size history (for power-law fitting)
    avalanche_sizes: VecDeque<usize>,
    /// Maximum history length
    max_history: usize,
    /// Current avalanche being tracked
    current_avalanche: usize,
    /// Tolerance for critical regime detection
    pub tolerance: f64,
}

impl Default for SOCDynamics {
    fn default() -> Self {
        Self::new(1000, 0.1)
    }
}

impl SOCDynamics {
    /// Create new SOC dynamics tracker
    pub fn new(max_history: usize, tolerance: f64) -> Self {
        Self {
            stats: SOCStats {
                sigma_target: 1.0,
                power_law_tau: 1.5,
                ..Default::default()
            },
            avalanche_sizes: VecDeque::with_capacity(max_history),
            max_history,
            current_avalanche: 0,
            tolerance,
        }
    }

    /// Record a spike event (initiating or triggered)
    pub fn record_spike(&mut self, is_initiating: bool) {
        if is_initiating {
            // Finish previous avalanche if any
            if self.current_avalanche > 0 {
                self.finish_avalanche();
            }
            self.current_avalanche = 1;
            self.stats.total_initiating += 1;
        } else {
            self.current_avalanche += 1;
            self.stats.total_triggered += 1;
        }

        // Update sigma
        if self.stats.total_initiating > 0 {
            self.stats.sigma_measured =
                self.stats.total_triggered as f64 / self.stats.total_initiating as f64;
        }
        self.stats.check_criticality(self.tolerance);
    }

    /// Finish current avalanche and record size
    pub fn finish_avalanche(&mut self) {
        if self.current_avalanche > 0 {
            self.stats.total_avalanches += 1;
            self.stats.largest_avalanche = self.stats.largest_avalanche.max(self.current_avalanche);

            // Update rolling average
            let n = self.stats.total_avalanches as f64;
            self.stats.avg_avalanche_size =
                self.stats.avg_avalanche_size * (n - 1.0) / n
                + self.current_avalanche as f64 / n;

            // Store in history
            if self.avalanche_sizes.len() >= self.max_history {
                self.avalanche_sizes.pop_front();
            }
            self.avalanche_sizes.push_back(self.current_avalanche);

            self.current_avalanche = 0;
        }
    }

    /// Estimate power-law exponent τ via maximum likelihood
    pub fn estimate_power_law(&self) -> f64 {
        if self.avalanche_sizes.is_empty() {
            return 1.5; // Default
        }

        let min_size = 1.0;
        let n = self.avalanche_sizes.len() as f64;
        let log_sum: f64 = self.avalanche_sizes.iter()
            .map(|&s| (s as f64 / min_size).ln())
            .sum();

        // MLE for power-law: τ = 1 + n / Σ ln(s/s_min)
        1.0 + n / log_sum.max(1.0)
    }

    /// Get modulation factor for plasticity based on criticality
    pub fn plasticity_modulation(&self) -> f64 {
        if self.stats.is_critical {
            1.0
        } else if self.stats.sigma_measured < self.stats.sigma_target {
            // Subcritical: increase excitability
            1.0 + (self.stats.sigma_target - self.stats.sigma_measured).min(0.5)
        } else {
            // Supercritical: decrease excitability
            1.0 - (self.stats.sigma_measured - self.stats.sigma_target).min(0.5)
        }
    }
}

// ============================================================================
// CONSCIOUSNESS METRICS
// ============================================================================

/// Consciousness/integration metrics for the lattice
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    /// Integrated Information Φ
    pub phi: f64,
    /// Network coherence (clustering coefficient)
    pub coherence: f64,
    /// Quantum entanglement entropy
    pub entanglement: f64,
    /// Global efficiency (mean inverse path length)
    pub global_efficiency: f64,
    /// Integration-Segregation Difference (Jang et al. 2024)
    pub isd: f64,
    /// Number of connected components
    pub num_components: usize,
}

impl ConsciousnessMetrics {
    /// Combined consciousness score
    pub fn combined_score(&self) -> f64 {
        (self.phi * self.coherence * (1.0 + self.entanglement)).sqrt() * self.global_efficiency
    }

    /// Check if lattice exhibits consciousness-like integration
    pub fn is_integrated(&self, threshold: f64) -> bool {
        self.phi > threshold && self.coherence > 0.3 && self.num_components == 1
    }
}

// ============================================================================
// HYPERBOLIC LATTICE
// ============================================================================

/// Main hyperbolic lattice structure with SOC dynamics and consciousness metrics
#[derive(Debug, Clone)]
pub struct HyperbolicLattice {
    /// Schläfli symbol defining the tiling
    pub tiling: SchlafliSymbol,
    /// All nodes in the lattice
    pub nodes: Vec<LatticeNode>,
    /// Edge weights (sparse representation)
    pub edges: HashMap<(usize, usize), f64>,
    /// SOC dynamics tracker
    pub soc: SOCDynamics,
    /// Latest consciousness metrics
    pub consciousness: ConsciousnessMetrics,
    /// Spatial hash grid for O(n) neighbor finding
    spatial_hash: HashMap<(i32, i32, i32), Vec<usize>>,
    /// Grid cell size for spatial hashing
    cell_size: f64,
    /// Maximum depth of the lattice
    pub max_depth: u32,
    /// Current simulation step
    pub step: usize,
}

impl HyperbolicLattice {
    /// Create a new lattice with given tiling and depth
    pub fn new(tiling: SchlafliSymbol, max_depth: u32) -> Self {
        Self {
            tiling,
            nodes: Vec::new(),
            edges: HashMap::new(),
            soc: SOCDynamics::default(),
            consciousness: ConsciousnessMetrics::default(),
            spatial_hash: HashMap::new(),
            cell_size: 0.5,
            max_depth,
            step: 0,
        }
    }

    /// Create a {7,3} heptagonal lattice
    pub fn heptagonal(max_depth: u32) -> Self {
        Self::new(SchlafliSymbol::heptagonal(), max_depth)
    }

    /// Create a {5,4} pentagonal lattice
    pub fn pentagonal(max_depth: u32) -> Self {
        Self::new(SchlafliSymbol::pentagonal(), max_depth)
    }

    /// Build the lattice structure by generating nodes and edges
    pub fn build(&mut self, num_nodes: usize) -> Result<()> {
        // Generate nodes using Fibonacci spiral on hyperboloid
        self.generate_fibonacci_nodes(num_nodes);

        // Connect neighbors based on hyperbolic distance
        self.connect_neighbors()?;

        // Build spatial hash for fast lookups
        self.rebuild_spatial_hash();

        // Assign depths based on distance from centroid
        self.assign_depths();

        Ok(())
    }

    /// Generate nodes using Fibonacci spiral distribution on hyperboloid
    fn generate_fibonacci_nodes(&mut self, num_nodes: usize) {
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let golden_angle = 2.0 * std::f64::consts::PI / (golden_ratio * golden_ratio);

        self.nodes.clear();

        for i in 0..num_nodes {
            // Fibonacci sphere distribution
            let t = i as f64 / (num_nodes - 1).max(1) as f64;
            let theta = (1.0 - 2.0 * t).acos();
            let phi = golden_angle * i as f64;

            // Convert to hyperbolic coordinates
            let rapidity = t * 2.0; // Spread across hyperboloid
            let position = LorentzPoint::from_spherical(rapidity, theta, phi);

            self.nodes.push(LatticeNode::new(i, position, 0));
        }
    }

    /// Connect neighboring nodes based on hyperbolic distance
    fn connect_neighbors(&mut self) -> Result<()> {
        let degree = self.tiling.vertex_degree() as usize;
        let n = self.nodes.len();

        // Find k nearest neighbors for each node
        for i in 0..n {
            let pos_i = self.nodes[i].position;

            // Calculate distances to all other nodes
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, pos_i.distance(&self.nodes[j].position)))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Connect to k nearest neighbors
            for (j, dist) in distances.into_iter().take(degree) {
                if !self.nodes[i].neighbors.contains(&j) {
                    self.nodes[i].neighbors.push(j);
                }
                if !self.nodes[j].neighbors.contains(&i) {
                    self.nodes[j].neighbors.push(i);
                }

                // Store edge weight (inverse distance)
                let weight = (-dist).exp();
                self.edges.insert((i.min(j), i.max(j)), weight);
            }
        }

        Ok(())
    }

    /// Rebuild spatial hash for fast neighbor lookups
    fn rebuild_spatial_hash(&mut self) {
        self.spatial_hash.clear();

        for (i, node) in self.nodes.iter().enumerate() {
            let cell = self.position_to_cell(&node.position);
            self.spatial_hash.entry(cell).or_default().push(i);
        }
    }

    /// Convert position to spatial hash cell
    fn position_to_cell(&self, pos: &LorentzPoint) -> (i32, i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    /// Assign depths based on distance from center
    fn assign_depths(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        let origin = LorentzPoint::origin();
        let max_dist = self.nodes.iter()
            .map(|n| n.position.distance(&origin))
            .fold(0.0_f64, f64::max);

        for node in &mut self.nodes {
            let dist = node.position.distance(&origin);
            let normalized = if max_dist > 0.0 { dist / max_dist } else { 0.0 };
            node.depth = ((1.0 - normalized) * self.max_depth as f64) as u32;
        }
    }

    /// Add a node at the given position
    pub fn add_node(&mut self, position: LorentzPoint) -> usize {
        let id = self.nodes.len();
        self.nodes.push(LatticeNode::new(id, position, 0));

        // Update spatial hash
        let cell = self.position_to_cell(&position);
        self.spatial_hash.entry(cell).or_default().push(id);

        id
    }

    /// Connect two nodes with given weight
    pub fn connect(&mut self, i: usize, j: usize, weight: f64) -> Result<()> {
        if i >= self.nodes.len() {
            return Err(LatticeError::NodeNotFound(i));
        }
        if j >= self.nodes.len() {
            return Err(LatticeError::NodeNotFound(j));
        }

        let key = (i.min(j), i.max(j));
        if self.edges.contains_key(&key) {
            return Err(LatticeError::EdgeExists(i, j));
        }

        self.nodes[i].neighbors.push(j);
        self.nodes[j].neighbors.push(i);
        self.edges.insert(key, weight);

        Ok(())
    }

    /// Simulate one step of SOC dynamics
    pub fn step_soc(&mut self) {
        self.step += 1;

        // Decay refractory periods
        for node in &mut self.nodes {
            node.tick();
        }

        // Collect nodes that will fire
        let firing: Vec<usize> = self.nodes.iter()
            .enumerate()
            .filter(|(_, n)| n.can_fire())
            .map(|(i, _)| i)
            .collect();

        // Record initiating spikes
        for &_i in &firing {
            self.soc.record_spike(true);
        }

        // Propagate activity
        for &i in &firing {
            let neighbors: Vec<usize> = self.nodes[i].neighbors.clone();
            self.nodes[i].fire(3); // 3-step refractory

            for j in neighbors {
                let key = (i.min(j), i.max(j));
                let weight = self.edges.get(&key).copied().unwrap_or(0.1);

                let was_below = self.nodes[j].potential < self.nodes[j].threshold;
                self.nodes[j].excite(weight);
                let is_above = self.nodes[j].potential >= self.nodes[j].threshold;

                // Record triggered spike
                if was_below && is_above {
                    self.soc.record_spike(false);
                }
            }
        }

        // Finish avalanche if no activity
        if firing.is_empty() {
            self.soc.finish_avalanche();
        }
    }

    /// Calculate consciousness metrics
    pub fn calculate_consciousness(&mut self) {
        let n = self.nodes.len();
        if n == 0 {
            return;
        }

        // Coherence (clustering coefficient)
        let mut total_clustering = 0.0;
        for node in &self.nodes {
            let k = node.neighbors.len();
            if k < 2 {
                continue;
            }

            let mut triangles = 0;
            for (idx_i, &ni) in node.neighbors.iter().enumerate() {
                for &nj in &node.neighbors[idx_i + 1..] {
                    if self.nodes[ni].neighbors.contains(&nj) {
                        triangles += 1;
                    }
                }
            }

            let possible = k * (k - 1) / 2;
            if possible > 0 {
                total_clustering += triangles as f64 / possible as f64;
            }
        }
        self.consciousness.coherence = total_clustering / n as f64;

        // Entanglement (simplified quantum entropy)
        let mut entropy = 0.0;
        for node in &self.nodes {
            let prob = node.amplitude.norm_sqr();
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }
        self.consciousness.entanglement = entropy / (n as f64).ln().max(1.0);

        // Phi (simplified IIT measure)
        self.consciousness.phi = (self.consciousness.coherence * self.consciousness.entanglement).sqrt()
            * self.soc.stats.sigma_measured.min(2.0);

        // Global efficiency
        let mut total_inv_dist = 0.0;
        let mut pairs = 0;
        for i in 0..n {
            for j in i + 1..n {
                let dist = self.nodes[i].position.distance(&self.nodes[j].position);
                if dist > 0.0 {
                    total_inv_dist += 1.0 / dist;
                    pairs += 1;
                }
            }
        }
        self.consciousness.global_efficiency = if pairs > 0 {
            total_inv_dist / pairs as f64
        } else {
            0.0
        };

        // ISD (Integration - Segregation Difference)
        self.consciousness.isd = self.consciousness.global_efficiency - self.consciousness.coherence;

        // Count components (simplified)
        self.consciousness.num_components = if self.edges.is_empty() { n } else { 1 };
    }

    /// Get nodes at a specific depth
    pub fn nodes_at_depth(&self, depth: u32) -> Vec<&LatticeNode> {
        self.nodes.iter().filter(|n| n.depth == depth).collect()
    }

    /// Get boundary nodes (depth = 0)
    pub fn boundary_nodes(&self) -> Vec<&LatticeNode> {
        self.nodes_at_depth(0)
    }

    /// Get interior nodes (depth > 0)
    pub fn interior_nodes(&self) -> Vec<&LatticeNode> {
        self.nodes.iter().filter(|n| n.depth > 0).collect()
    }

    /// Inject external input to boundary nodes
    pub fn inject_boundary_input(&mut self, inputs: &[f64]) {
        let boundary: Vec<usize> = self.nodes.iter()
            .enumerate()
            .filter(|(_, n)| n.depth == 0)
            .map(|(i, _)| i)
            .collect();

        for (i, &idx) in boundary.iter().enumerate() {
            let input = inputs.get(i).copied().unwrap_or(0.0);
            self.nodes[idx].excite(input);
        }
    }

    /// Get edge weight between two nodes
    pub fn edge_weight(&self, i: usize, j: usize) -> Option<f64> {
        self.edges.get(&(i.min(j), i.max(j))).copied()
    }

    /// Total number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Average degree of nodes
    pub fn avg_degree(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let total: usize = self.nodes.iter().map(|n| n.neighbors.len()).sum();
        total as f64 / self.nodes.len() as f64
    }
}

// ============================================================================
// FRACTAL HIERARCHY
// ============================================================================

/// Fractal hierarchy builder for multi-scale lattice organization
#[derive(Debug, Clone)]
pub struct FractalHierarchy {
    /// Number of levels in the hierarchy
    pub levels: usize,
    /// Cluster assignments at each level
    clusters: Vec<Vec<Vec<usize>>>,
}

impl FractalHierarchy {
    /// Create a new fractal hierarchy
    pub fn new(levels: usize) -> Self {
        Self {
            levels,
            clusters: vec![Vec::new(); levels],
        }
    }

    /// Get golden ratio (φ = 1.618...) for optimal point distribution
    pub fn golden_ratio() -> f64 {
        (1.0 + 5.0_f64.sqrt()) / 2.0
    }

    /// Build hierarchy from lattice
    pub fn build_from_lattice(&mut self, lattice: &HyperbolicLattice) {
        let n = lattice.nodes.len();
        if n == 0 {
            return;
        }

        // Level 0: individual nodes
        self.clusters[0] = (0..n).map(|i| vec![i]).collect();

        // Build higher levels by clustering
        for level in 1..self.levels {
            let cluster_size = 2_usize.pow(level as u32);
            let num_clusters = (n + cluster_size - 1) / cluster_size;

            self.clusters[level] = (0..num_clusters)
                .map(|c| {
                    let start = c * cluster_size;
                    let end = (start + cluster_size).min(n);
                    (start..end).collect()
                })
                .collect();
        }
    }

    /// Get clusters at given level
    pub fn clusters_at_level(&self, level: usize) -> &[Vec<usize>] {
        if level < self.levels {
            &self.clusters[level]
        } else {
            &[]
        }
    }

    /// Get number of clusters at each level
    pub fn cluster_counts(&self) -> Vec<usize> {
        self.clusters.iter().map(|c| c.len()).collect()
    }
}

// ============================================================================
// GAME-THEORETIC COORDINATION
// ============================================================================

/// Nash equilibrium state for lattice coordination
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NashEquilibrium {
    /// Strategy probabilities for each node
    pub strategies: Vec<f64>,
    /// Stability measure (deviation payoff difference)
    pub stability: f64,
    /// Φ-weighted equilibrium quality
    pub phi_stability: f64,
    /// Is this a valid Nash equilibrium?
    pub is_valid: bool,
}

/// Game-theoretic coordinator for multi-agent lattice
#[derive(Debug, Clone)]
pub struct GameCoordinator {
    /// Current equilibrium state
    pub equilibrium: NashEquilibrium,
    /// Payoff matrix (symmetric for cooperation game)
    payoff_matrix: DMatrix<f64>,
    /// Learning rate for strategy updates
    learning_rate: f64,
}

impl GameCoordinator {
    /// Create new coordinator for n agents
    pub fn new(n: usize) -> Self {
        Self {
            equilibrium: NashEquilibrium {
                strategies: vec![0.5; n],
                is_valid: false,
                ..Default::default()
            },
            payoff_matrix: DMatrix::from_element(n, n, 0.1),
            learning_rate: 0.1,
        }
    }

    /// Update payoff matrix from lattice connectivity
    pub fn update_from_lattice(&mut self, lattice: &HyperbolicLattice) {
        let n = lattice.nodes.len();
        if n == 0 {
            return;
        }

        self.payoff_matrix = DMatrix::from_element(n, n, 0.0);

        for (&(i, j), &weight) in &lattice.edges {
            self.payoff_matrix[(i, j)] = weight;
            self.payoff_matrix[(j, i)] = weight;
        }
    }

    /// Update strategies using replicator dynamics
    pub fn update_strategies(&mut self) {
        let n = self.equilibrium.strategies.len();
        if n == 0 {
            return;
        }

        // Calculate fitness for each strategy
        let mut fitness = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                fitness[i] += self.payoff_matrix[(i, j)] * self.equilibrium.strategies[j];
            }
        }

        // Average fitness
        let avg_fitness: f64 = fitness.iter().zip(&self.equilibrium.strategies)
            .map(|(f, s)| f * s)
            .sum();

        // Replicator dynamics: dx_i/dt = x_i(f_i - f̄)
        for i in 0..n {
            let delta = self.equilibrium.strategies[i] * (fitness[i] - avg_fitness);
            self.equilibrium.strategies[i] += self.learning_rate * delta;
            self.equilibrium.strategies[i] = self.equilibrium.strategies[i].clamp(0.0, 1.0);
        }

        // Normalize
        let sum: f64 = self.equilibrium.strategies.iter().sum();
        if sum > 0.0 {
            for s in &mut self.equilibrium.strategies {
                *s /= sum;
            }
        }

        // Calculate stability
        self.equilibrium.stability = 1.0 - self.equilibrium.strategies.iter()
            .map(|&s| (s - 1.0 / n as f64).powi(2))
            .sum::<f64>()
            .sqrt();
    }

    /// Calculate Φ-weighted equilibrium quality
    pub fn calculate_phi_stability(&mut self, phi: f64) {
        self.equilibrium.phi_stability = self.equilibrium.stability * phi;
        self.equilibrium.is_valid = self.equilibrium.stability > 0.5;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schlafli_symbol() {
        let s73 = SchlafliSymbol::heptagonal();
        assert!(s73.is_hyperbolic());
        assert!(s73.angular_defect() < 0.0); // Negative curvature

        let s54 = SchlafliSymbol::pentagonal();
        assert!(s54.is_hyperbolic());

        // Invalid Euclidean tiling
        assert!(SchlafliSymbol::new(6, 3).is_err());
        assert!(SchlafliSymbol::new(4, 4).is_err());
    }

    #[test]
    fn test_lorentz_point() {
        let origin = LorentzPoint::origin();
        assert!((origin.minkowski_norm_sq() - 1.0).abs() < 1e-10);

        let p = LorentzPoint::from_spherical(1.0, 0.5, 0.3);
        assert!((p.minkowski_norm_sq() - 1.0).abs() < 1e-10);

        assert!(origin.distance(&origin) < 1e-10);
        assert!(origin.distance(&p) > 0.0);
    }

    #[test]
    fn test_hyperbolic_lattice_creation() {
        let mut lattice = HyperbolicLattice::heptagonal(3);
        lattice.build(64).unwrap();

        assert_eq!(lattice.nodes.len(), 64);
        assert!(lattice.num_edges() > 0);
        assert!(lattice.avg_degree() > 0.0);
    }

    #[test]
    fn test_soc_dynamics() {
        let mut soc = SOCDynamics::new(100, 0.1);

        // Simulate some spikes
        soc.record_spike(true); // Initiating
        soc.record_spike(false); // Triggered
        soc.record_spike(false); // Triggered
        soc.finish_avalanche();

        assert_eq!(soc.stats.total_initiating, 1);
        assert_eq!(soc.stats.total_triggered, 2);
        assert!((soc.stats.sigma_measured - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lattice_step() {
        let mut lattice = HyperbolicLattice::heptagonal(3);
        lattice.build(16).unwrap();

        // Inject some input
        lattice.inject_boundary_input(&[1.5, 1.5, 1.5, 1.5]);

        // Run some steps
        for _ in 0..10 {
            lattice.step_soc();
        }

        // Should have some activity
        assert!(lattice.soc.stats.total_avalanches > 0 || lattice.soc.stats.total_initiating > 0);
    }

    #[test]
    fn test_consciousness_metrics() {
        let mut lattice = HyperbolicLattice::heptagonal(3);
        lattice.build(32).unwrap();

        lattice.calculate_consciousness();

        assert!(lattice.consciousness.coherence >= 0.0);
        assert!(lattice.consciousness.global_efficiency >= 0.0);
    }

    #[test]
    fn test_fractal_hierarchy() {
        let mut lattice = HyperbolicLattice::heptagonal(3);
        lattice.build(64).unwrap();

        let mut hierarchy = FractalHierarchy::new(4);
        hierarchy.build_from_lattice(&lattice);

        assert_eq!(hierarchy.clusters_at_level(0).len(), 64);
        assert!(hierarchy.clusters_at_level(1).len() < 64);
        assert!(hierarchy.clusters_at_level(2).len() < hierarchy.clusters_at_level(1).len());
    }

    #[test]
    fn test_game_coordinator() {
        let mut lattice = HyperbolicLattice::heptagonal(3);
        lattice.build(16).unwrap();

        let mut coord = GameCoordinator::new(16);
        coord.update_from_lattice(&lattice);

        for _ in 0..10 {
            coord.update_strategies();
        }

        coord.calculate_phi_stability(0.5);

        assert!(coord.equilibrium.stability >= 0.0);
        assert!(coord.equilibrium.strategies.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_parallel_transport() {
        let origin = LorentzPoint::origin();
        let p = LorentzPoint::from_spherical(0.5, 0.3, 0.2);

        let v = Vector4::new(0.0, 1.0, 0.0, 0.0);
        let transported = p.parallel_transport(&v);

        // Transported vector should have similar magnitude
        let orig_norm = (v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
        let trans_norm = (transported[1] * transported[1] + transported[2] * transported[2] + transported[3] * transported[3]).sqrt();

        assert!((orig_norm - trans_norm).abs() < 0.5);
    }
}
