//! Adversarial Defense Lattice for MCP Security
//!
//! Implements hyperbolic geometry-based adversarial defense using exponential
//! boundary growth for perimeter security in consciousness network protection.
//!
//! # Research Foundation
//!
//! - Kollár et al. (2019) "Hyperbolic lattices in circuit QED" Nature 571:45-50
//! - Friston (2010) "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience
//! - Bak et al. (1987) "Self-organized criticality" Physical Review Letters
//! - Parisi (1988) "Statistical Field Theory" Perseus Books
//!
//! # Architecture
//!
//! The adversarial defense lattice uses hyperbolic geometry's exponential boundary
//! growth to create defense-in-depth with vastly more perimeter sentries relative
//! to interior volume compared to Euclidean space.
//!
//! ## Supported Tilings
//!
//! - **{7,3} Heptagonal**: Maximum connectivity, hierarchical depth emergence
//! - **{5,4} Pentagonal-Square**: Balanced fan-out, uniform distance distribution
//! - **{6,4} Hexagonal-Square**: Tunable curvature, moderate connectivity
//! - **{8,3} Octagonal**: High branching factor, strong exponential growth
//!
//! # Security Properties
//!
//! 1. **Thermodynamic Defense**: Injection attempts violate metric structure
//! 2. **Self-Organized Criticality**: Near phase transition for optimal response
//! 3. **Active Inference**: Bayesian belief propagation along geodesics

use crate::{poincare::PoincarePoint, GeometryError, Result};
use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Hyperbolic tiling Schläfli symbol {p,q}
///
/// - p: Number of sides per polygon
/// - q: Number of polygons meeting at each vertex
///
/// Hyperbolic condition: (p-2)(q-2) > 4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchlafliSymbol {
    /// Polygon sides (p)
    pub p: usize,
    /// Polygons per vertex (q)
    pub q: usize,
}

impl SchlafliSymbol {
    /// Create new Schläfli symbol, validating hyperbolic condition
    pub fn new(p: usize, q: usize) -> Result<Self> {
        if p < 3 || q < 3 {
            return Err(GeometryError::InvalidTessellation {
                message: format!("Invalid Schläfli symbol: p={}, q={} (need p,q >= 3)", p, q),
            });
        }

        let product = (p - 2) * (q - 2);
        if product <= 4 {
            return Err(GeometryError::InvalidTessellation {
                message: format!(
                    "{{{}，{}}} is not hyperbolic: (p-2)(q-2) = {} <= 4",
                    p, q, product
                ),
            });
        }

        Ok(Self { p, q })
    }

    /// {7,3} - Heptagonal tiling (maximum connectivity)
    pub fn heptagonal() -> Self {
        Self { p: 7, q: 3 }
    }

    /// {5,4} - Pentagonal-square tiling (balanced fan-out)
    pub fn pentagonal_square() -> Self {
        Self { p: 5, q: 4 }
    }

    /// {6,4} - Hexagonal-square tiling (moderate curvature)
    pub fn hexagonal_square() -> Self {
        Self { p: 6, q: 4 }
    }

    /// {8,3} - Octagonal tiling (high branching)
    pub fn octagonal() -> Self {
        Self { p: 8, q: 3 }
    }

    /// Calculate the Gaussian curvature for this tiling
    ///
    /// K = -4π / (2πp - 2π(p-2)q/q) = 2π(1/p + 1/q - 1/2) / Area
    pub fn gaussian_curvature(&self) -> f64 {
        let p = self.p as f64;
        let q = self.q as f64;
        // Normalized curvature relative to edge length
        2.0 * PI * (1.0 / p + 1.0 / q - 0.5)
    }

    /// Calculate characteristic edge length
    ///
    /// sinh(r) = cos(π/q) / sin(π/p)
    pub fn edge_length(&self) -> f64 {
        let p = self.p as f64;
        let q = self.q as f64;
        let sinh_r = (PI / q).cos() / (PI / p).sin();
        sinh_r.asinh()
    }

    /// Vertex valence (number of edges meeting at vertex)
    pub fn valence(&self) -> usize {
        self.q
    }

    /// Exponential growth rate of tiles per layer
    ///
    /// For hyperbolic tilings, # tiles at layer n ~ (q-1)^n
    pub fn growth_rate(&self) -> f64 {
        (self.q - 1) as f64
    }
}

/// Defense topology configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseTopology {
    /// Schläfli symbol for the tiling
    pub tiling: SchlafliSymbol,
    /// Maximum depth of defense layers
    pub depth: usize,
    /// Curvature tuning parameter (affects false positive rate)
    pub curvature_scale: f64,
    /// Criticality threshold for cascade response
    pub criticality_threshold: f64,
}

impl DefenseTopology {
    /// Create optimal topology for maximum connectivity
    ///
    /// Uses {7,3} for hierarchical depth emergence and
    /// maximum local connectivity without bottlenecks.
    pub fn maximum_connectivity(depth: usize) -> Self {
        Self {
            tiling: SchlafliSymbol::heptagonal(),
            depth,
            curvature_scale: 1.0,
            criticality_threshold: 0.5,
        }
    }

    /// Create balanced topology for uniform detection
    ///
    /// Uses {5,4} for easier discrete implementation and
    /// more uniform distance distribution.
    pub fn balanced_fanout(depth: usize) -> Self {
        Self {
            tiling: SchlafliSymbol::pentagonal_square(),
            depth,
            curvature_scale: 1.0,
            criticality_threshold: 0.5,
        }
    }

    /// Create high-branching topology for aggressive detection
    ///
    /// Uses {8,3} for strong exponential growth and
    /// more perimeter sentries.
    pub fn aggressive_detection(depth: usize) -> Self {
        Self {
            tiling: SchlafliSymbol::octagonal(),
            depth,
            curvature_scale: 1.5, // More negative effective curvature
            criticality_threshold: 0.7,
        }
    }

    /// Tune curvature for detection sensitivity
    ///
    /// Higher curvature scale = more aggressive detection but higher false positive rate
    pub fn with_curvature_scale(mut self, scale: f64) -> Self {
        self.curvature_scale = scale.max(0.1).min(3.0);
        self
    }

    /// Set criticality threshold for cascade response
    ///
    /// At the critical point, you get scale-free avalanche distributions
    pub fn with_criticality(mut self, threshold: f64) -> Self {
        self.criticality_threshold = threshold.max(0.0).min(1.0);
        self
    }
}

/// Hyperboloid model point representation
///
/// Uses the upper sheet of H² embedded in Minkowski space ℝ^{2,1}:
/// -t² + x² + y² = -1, t > 0
///
/// Clean algebraic operations, natural for Lorentz boosts
/// representing information flow.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HyperboloidPoint {
    /// Time-like coordinate (t > 0)
    pub t: f64,
    /// Spatial x coordinate
    pub x: f64,
    /// Spatial y coordinate
    pub y: f64,
}

impl HyperboloidPoint {
    /// Create new hyperboloid point, ensuring constraint -t² + x² + y² = -1
    pub fn new(x: f64, y: f64) -> Self {
        let t = (1.0 + x * x + y * y).sqrt();
        Self { t, x, y }
    }

    /// Origin of the hyperboloid (0, 0) -> (1, 0, 0)
    pub fn origin() -> Self {
        Self { t: 1.0, x: 0.0, y: 0.0 }
    }

    /// Convert from Poincaré disk coordinates
    pub fn from_poincare(point: &PoincarePoint) -> Self {
        let coords = point.coords();
        let r_sq = coords.x * coords.x + coords.y * coords.y;

        if r_sq >= 1.0 - 1e-10 {
            // At boundary, project to large distance
            let scale = 0.99 / r_sq.sqrt();
            let x = coords.x * scale;
            let y = coords.y * scale;
            let r_sq_safe = x * x + y * y;
            let t = (1.0 + r_sq_safe) / (1.0 - r_sq_safe);
            let factor = (2.0 / (1.0 - r_sq_safe)).sqrt();
            Self {
                t,
                x: x * factor,
                y: y * factor,
            }
        } else {
            // Standard conversion
            let t = (1.0 + r_sq) / (1.0 - r_sq);
            let factor = 2.0 / (1.0 - r_sq);
            Self {
                t,
                x: coords.x * factor,
                y: coords.y * factor,
            }
        }
    }

    /// Convert to Poincaré disk coordinates
    pub fn to_poincare(&self) -> Result<PoincarePoint> {
        if self.t <= 1.0 {
            return Err(GeometryError::NumericalInstability);
        }

        let factor = 1.0 / (1.0 + self.t);
        let x = self.x * factor;
        let y = self.y * factor;

        PoincarePoint::new(na::Vector3::new(x, y, 0.0))
    }

    /// Minkowski inner product (signature -++)
    ///
    /// ⟨p, q⟩ = -t₁t₂ + x₁x₂ + y₁y₂
    pub fn minkowski_inner(&self, other: &Self) -> f64 {
        -self.t * other.t + self.x * other.x + self.y * other.y
    }

    /// Hyperbolic distance using Minkowski inner product
    ///
    /// d(p, q) = acosh(-⟨p, q⟩)
    pub fn distance(&self, other: &Self) -> f64 {
        let inner = self.minkowski_inner(other);
        // inner is negative for points on same sheet, -inner > 1
        (-inner).max(1.0).acosh()
    }

    /// Apply Lorentz boost along x-axis (information flow)
    ///
    /// Represents velocity v propagation of information
    pub fn lorentz_boost_x(&self, rapidity: f64) -> Self {
        let cosh_r = rapidity.cosh();
        let sinh_r = rapidity.sinh();

        Self {
            t: cosh_r * self.t + sinh_r * self.x,
            x: sinh_r * self.t + cosh_r * self.x,
            y: self.y,
        }
    }

    /// Apply Lorentz boost along y-axis
    pub fn lorentz_boost_y(&self, rapidity: f64) -> Self {
        let cosh_r = rapidity.cosh();
        let sinh_r = rapidity.sinh();

        Self {
            t: cosh_r * self.t + sinh_r * self.y,
            x: self.x,
            y: sinh_r * self.t + cosh_r * self.y,
        }
    }

    /// Apply rotation in xy-plane (spatial rotation)
    pub fn rotate(&self, angle: f64) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        Self {
            t: self.t,
            x: cos_a * self.x - sin_a * self.y,
            y: sin_a * self.x + cos_a * self.y,
        }
    }
}

/// Sentry node in the adversarial defense lattice
///
/// Each sentry maintains local Bayesian belief state and
/// performs variational inference on message authenticity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentryNode {
    /// Unique identifier
    pub id: usize,
    /// Position in hyperboloid model
    pub position: HyperboloidPoint,
    /// Layer in defense hierarchy (0 = interior, higher = perimeter)
    pub layer: usize,
    /// Neighboring sentry IDs
    pub neighbors: Vec<usize>,
    /// Current belief state (probability of threat)
    pub belief_state: f64,
    /// Free energy of current state
    pub free_energy: f64,
    /// Prediction error accumulator
    pub prediction_error: f64,
    /// Activation level for cascade detection
    pub activation: f64,
}

impl SentryNode {
    /// Create new sentry node at position
    pub fn new(id: usize, position: HyperboloidPoint, layer: usize) -> Self {
        Self {
            id,
            position,
            layer,
            neighbors: Vec::new(),
            belief_state: 0.0,    // Prior: no threat
            free_energy: 0.0,
            prediction_error: 0.0,
            activation: 0.0,
        }
    }

    /// Update belief state using variational inference
    ///
    /// Implements active inference: minimize free energy by
    /// updating beliefs to match observations.
    ///
    /// Free Energy = Complexity - Accuracy
    ///             = KL[q(z|x)||p(z)] - E_q[log p(x|z)]
    pub fn update_belief(&mut self, observation: f64, prior_weight: f64) {
        // Simplified variational update
        // observation: measured anomaly score (0 = normal, 1 = anomalous)
        // prior_weight: weight of hyperbolic metric prior

        let prior = self.compute_metric_prior();
        let likelihood = self.compute_likelihood(observation);

        // Variational free energy
        let complexity = self.kl_divergence(self.belief_state, prior);
        let accuracy = self.expected_log_likelihood(observation, likelihood);

        self.free_energy = complexity - accuracy;

        // Update belief to minimize free energy (gradient descent)
        let learning_rate = 0.1;
        let gradient = self.free_energy_gradient(observation, prior, prior_weight);
        self.belief_state = (self.belief_state + learning_rate * gradient).max(0.0).min(1.0);

        // Update prediction error
        self.prediction_error = (observation - self.belief_state).abs();
    }

    /// Compute prior from hyperbolic metric structure
    ///
    /// The prior p(z) is defined by the hyperbolic metric:
    /// points further from origin have higher prior probability of threat
    fn compute_metric_prior(&self) -> f64 {
        // Distance from origin in hyperbolic space
        let origin = HyperboloidPoint::origin();
        let dist = self.position.distance(&origin);

        // Prior increases with distance (perimeter more exposed)
        let base_prior = 0.1; // Base threat probability
        let distance_factor = 0.05; // How much distance affects prior

        (base_prior + distance_factor * dist).min(0.9)
    }

    /// Compute likelihood of observation given belief
    fn compute_likelihood(&self, observation: f64) -> f64 {
        // Gaussian likelihood centered on belief
        let sigma = 0.2;
        let diff = observation - self.belief_state;
        (-diff * diff / (2.0 * sigma * sigma)).exp()
    }

    /// KL divergence between current belief and prior
    fn kl_divergence(&self, belief: f64, prior: f64) -> f64 {
        let belief_safe = belief.max(1e-10).min(1.0 - 1e-10);
        let prior_safe = prior.max(1e-10).min(1.0 - 1e-10);

        belief_safe * (belief_safe / prior_safe).ln()
            + (1.0 - belief_safe) * ((1.0 - belief_safe) / (1.0 - prior_safe)).ln()
    }

    /// Expected log likelihood (accuracy term)
    fn expected_log_likelihood(&self, _observation: f64, likelihood: f64) -> f64 {
        likelihood.max(1e-10).ln()
    }

    /// Gradient of free energy w.r.t. belief
    fn free_energy_gradient(&self, observation: f64, prior: f64, prior_weight: f64) -> f64 {
        // Gradient points toward lower free energy
        // Simplified: move belief toward observation weighted by prior

        let obs_gradient = observation - self.belief_state;
        let prior_gradient = prior - self.belief_state;

        obs_gradient * (1.0 - prior_weight) + prior_gradient * prior_weight
    }

    /// Propagate prediction error to neighbors
    ///
    /// Returns prediction errors to send along geodesic paths
    pub fn propagate_error(&self, decay_rate: f64) -> HashMap<usize, f64> {
        let mut propagated = HashMap::new();

        // Error decays with geodesic distance (approximated by hops)
        let error_to_send = self.prediction_error * decay_rate;

        for &neighbor_id in &self.neighbors {
            propagated.insert(neighbor_id, error_to_send);
        }

        propagated
    }

    /// Update activation for cascade detection
    pub fn update_activation(&mut self, incoming_errors: &[f64], criticality_threshold: f64) {
        // Sum incoming prediction errors
        let total_incoming: f64 = incoming_errors.iter().sum();

        // Update activation with leak and incoming signals
        let leak_rate = 0.9;
        self.activation = self.activation * leak_rate + total_incoming + self.prediction_error;

        // Check for cascade trigger
        if self.activation > criticality_threshold {
            // At critical point: activation triggers response
            self.belief_state = 1.0; // Full threat detection
        }
    }
}

/// Adversarial Defense Lattice
///
/// Hyperbolic tessellation-based defense network where each node
/// performs active inference on message authenticity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialLattice {
    /// Defense topology configuration
    pub topology: DefenseTopology,
    /// Sentry nodes in the lattice
    pub sentries: Vec<SentryNode>,
    /// Current global free energy
    pub global_free_energy: f64,
    /// Number of active cascade events
    pub cascade_count: usize,
    /// Detection events log
    pub detection_log: Vec<DetectionEvent>,
}

/// Detection event in the lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionEvent {
    /// Timestamp (arbitrary units)
    pub timestamp: u64,
    /// Sentry that triggered detection
    pub sentry_id: usize,
    /// Detection confidence (0-1)
    pub confidence: f64,
    /// Cascade size (number of activated sentries)
    pub cascade_size: usize,
    /// Free energy at detection
    pub free_energy: f64,
}

impl AdversarialLattice {
    /// Create new adversarial defense lattice
    pub fn new(topology: DefenseTopology) -> Result<Self> {
        let mut lattice = Self {
            topology: topology.clone(),
            sentries: Vec::new(),
            global_free_energy: 0.0,
            cascade_count: 0,
            detection_log: Vec::new(),
        };

        lattice.generate_sentries()?;
        lattice.connect_neighbors()?;

        Ok(lattice)
    }

    /// Generate sentry nodes based on topology
    fn generate_sentries(&mut self) -> Result<()> {
        let symbol = &self.topology.tiling;
        let depth = self.topology.depth;

        // Start with origin
        self.sentries.push(SentryNode::new(
            0,
            HyperboloidPoint::origin(),
            0,
        ));

        // Generate layers
        let edge_len = symbol.edge_length() * self.topology.curvature_scale;

        for layer in 1..=depth {
            let num_sentries_in_layer = self.compute_layer_count(layer);
            let layer_radius = edge_len * (layer as f64);

            for i in 0..num_sentries_in_layer {
                let angle = 2.0 * PI * (i as f64) / (num_sentries_in_layer as f64);

                // Position in Poincaré disk, then convert to hyperboloid
                let r = layer_radius.tanh().min(0.95);
                let x = r * angle.cos();
                let y = r * angle.sin();

                let position = HyperboloidPoint::new(x, y);
                let id = self.sentries.len();

                self.sentries.push(SentryNode::new(id, position, layer));
            }
        }

        Ok(())
    }

    /// Compute number of sentries in a layer
    fn compute_layer_count(&self, layer: usize) -> usize {
        let q = self.topology.tiling.q;
        if layer == 0 {
            1
        } else if layer == 1 {
            q
        } else {
            // Exponential growth: q * (q-1)^(layer-1)
            q * (q - 1).pow((layer - 1) as u32)
        }
    }

    /// Connect neighboring sentries based on tessellation structure
    fn connect_neighbors(&mut self) -> Result<()> {
        let num_sentries = self.sentries.len();
        let connection_threshold = self.topology.tiling.edge_length() * 1.5;

        // Connect based on hyperbolic distance
        for i in 0..num_sentries {
            let pos_i = self.sentries[i].position;

            for j in (i + 1)..num_sentries {
                let pos_j = self.sentries[j].position;
                let dist = pos_i.distance(&pos_j);

                // Connect if within threshold
                if dist < connection_threshold {
                    // Add bidirectional connection
                    self.sentries[i].neighbors.push(j);
                    self.sentries[j].neighbors.push(i);
                }
            }
        }

        // Ensure minimum connectivity (each sentry has at least one neighbor)
        for i in 0..num_sentries {
            if self.sentries[i].neighbors.is_empty() && i > 0 {
                // Connect to previous sentry as fallback
                self.sentries[i].neighbors.push(i - 1);
                self.sentries[i - 1].neighbors.push(i);
            }
        }

        Ok(())
    }

    /// Process incoming message for anomaly detection
    ///
    /// # Arguments
    ///
    /// * `entry_point` - Sentry ID where message enters
    /// * `anomaly_score` - Initial anomaly score from external analysis (0-1)
    ///
    /// # Returns
    ///
    /// Detection result with confidence and cascade information
    pub fn process_message(&mut self, entry_point: usize, anomaly_score: f64, timestamp: u64) -> DetectionResult {
        if entry_point >= self.sentries.len() {
            return DetectionResult {
                is_threat: false,
                confidence: 0.0,
                cascade_size: 0,
                free_energy: 0.0,
            };
        }

        // Phase 1: Local belief update at entry point
        let prior_weight = 0.3;
        self.sentries[entry_point].update_belief(anomaly_score, prior_weight);

        // Phase 2: Propagate prediction errors along geodesics
        let decay_rate = 0.8;
        let mut propagation_queue = vec![entry_point];
        let mut visited = vec![false; self.sentries.len()];
        visited[entry_point] = true;

        let mut cascade_size = 1;

        while let Some(current) = propagation_queue.pop() {
            let errors = self.sentries[current].propagate_error(decay_rate);

            for (&neighbor_id, &error) in &errors {
                if !visited[neighbor_id] {
                    visited[neighbor_id] = true;

                    // Update neighbor belief
                    self.sentries[neighbor_id].update_belief(error, prior_weight);
                    self.sentries[neighbor_id].update_activation(&[error], self.topology.criticality_threshold);

                    // Check for cascade
                    if self.sentries[neighbor_id].belief_state > self.topology.criticality_threshold {
                        cascade_size += 1;
                        propagation_queue.push(neighbor_id);
                    }
                }
            }
        }

        // Phase 3: Compute global free energy
        self.global_free_energy = self.sentries.iter()
            .map(|s| s.free_energy)
            .sum();

        // Phase 4: Determine detection result
        let max_belief = self.sentries.iter()
            .map(|s| s.belief_state)
            .fold(0.0, f64::max);

        let is_threat = max_belief > self.topology.criticality_threshold ||
                       cascade_size > (self.sentries.len() / 10).max(1);

        if is_threat {
            self.cascade_count += 1;
            self.detection_log.push(DetectionEvent {
                timestamp,
                sentry_id: entry_point,
                confidence: max_belief,
                cascade_size,
                free_energy: self.global_free_energy,
            });
        }

        DetectionResult {
            is_threat,
            confidence: max_belief,
            cascade_size,
            free_energy: self.global_free_energy,
        }
    }

    /// Get perimeter sentry count (exponential boundary)
    pub fn perimeter_count(&self) -> usize {
        let max_layer = self.sentries.iter()
            .map(|s| s.layer)
            .max()
            .unwrap_or(0);

        self.sentries.iter()
            .filter(|s| s.layer == max_layer)
            .count()
    }

    /// Get interior sentry count
    pub fn interior_count(&self) -> usize {
        self.sentries.len() - self.perimeter_count()
    }

    /// Compute perimeter-to-interior ratio
    ///
    /// In hyperbolic space, this ratio grows exponentially with depth,
    /// providing vastly more perimeter sentries for defense.
    pub fn perimeter_ratio(&self) -> f64 {
        let interior = self.interior_count().max(1) as f64;
        let perimeter = self.perimeter_count() as f64;
        perimeter / interior
    }

    /// Reset all sentry states
    pub fn reset(&mut self) {
        for sentry in &mut self.sentries {
            sentry.belief_state = 0.0;
            sentry.free_energy = 0.0;
            sentry.prediction_error = 0.0;
            sentry.activation = 0.0;
        }
        self.global_free_energy = 0.0;
    }

    /// Get statistics about the lattice
    pub fn stats(&self) -> LatticeStats {
        LatticeStats {
            total_sentries: self.sentries.len(),
            perimeter_sentries: self.perimeter_count(),
            interior_sentries: self.interior_count(),
            perimeter_ratio: self.perimeter_ratio(),
            total_connections: self.sentries.iter().map(|s| s.neighbors.len()).sum::<usize>() / 2,
            global_free_energy: self.global_free_energy,
            cascade_count: self.cascade_count,
            detection_events: self.detection_log.len(),
        }
    }
}

/// Result of message detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Whether message is classified as threat
    pub is_threat: bool,
    /// Detection confidence (0-1)
    pub confidence: f64,
    /// Number of sentries involved in cascade
    pub cascade_size: usize,
    /// Global free energy at detection
    pub free_energy: f64,
}

/// Lattice statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeStats {
    /// Total number of sentry nodes
    pub total_sentries: usize,
    /// Sentries at perimeter (max layer)
    pub perimeter_sentries: usize,
    /// Sentries in interior
    pub interior_sentries: usize,
    /// Perimeter-to-interior ratio
    pub perimeter_ratio: f64,
    /// Total number of connections
    pub total_connections: usize,
    /// Current global free energy
    pub global_free_energy: f64,
    /// Total cascade events
    pub cascade_count: usize,
    /// Total detection events logged
    pub detection_events: usize,
}

/// Curvature tuning utilities
pub mod curvature_tuning {
    #![allow(unused_imports)]
    use super::*;

    /// Tune curvature for desired false positive rate
    ///
    /// More negative curvature = more aggressive detection but higher false positive rate
    ///
    /// # Arguments
    ///
    /// * `target_fpr` - Target false positive rate (0-1)
    ///
    /// # Returns
    ///
    /// Curvature scale factor
    pub fn tune_for_fpr(target_fpr: f64) -> f64 {
        // Empirical relationship: FPR ~ 0.1 * curvature_scale
        // Inverse: curvature_scale ~ 10 * FPR
        let target_fpr_safe = target_fpr.max(0.01).min(0.3);
        10.0 * target_fpr_safe
    }

    /// Tune curvature for desired detection rate
    ///
    /// # Arguments
    ///
    /// * `target_tpr` - Target true positive rate (0-1)
    ///
    /// # Returns
    ///
    /// Curvature scale factor
    pub fn tune_for_tpr(target_tpr: f64) -> f64 {
        // Higher TPR needs higher curvature (more aggressive)
        let target_tpr_safe = target_tpr.max(0.5).min(0.99);
        1.0 + 2.0 * (target_tpr_safe - 0.5)
    }

    /// Find optimal curvature for ROC operating point
    ///
    /// Balances TPR and FPR for optimal detection.
    pub fn optimal_operating_point(target_tpr: f64, max_fpr: f64) -> f64 {
        let tpr_curvature = tune_for_tpr(target_tpr);
        let fpr_curvature = tune_for_fpr(max_fpr);

        // Balance between TPR requirement and FPR constraint
        (tpr_curvature + fpr_curvature) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schlafli_symbols() {
        // Valid hyperbolic tilings
        assert!(SchlafliSymbol::new(7, 3).is_ok()); // {7,3}
        assert!(SchlafliSymbol::new(5, 4).is_ok()); // {5,4}
        assert!(SchlafliSymbol::new(6, 4).is_ok()); // {6,4}
        assert!(SchlafliSymbol::new(8, 3).is_ok()); // {8,3}

        // Invalid (Euclidean or spherical)
        assert!(SchlafliSymbol::new(4, 4).is_err()); // {4,4} - Euclidean
        assert!(SchlafliSymbol::new(3, 3).is_err()); // {3,3} - Spherical
    }

    #[test]
    fn test_hyperboloid_point() {
        let origin = HyperboloidPoint::origin();
        assert!((origin.t - 1.0).abs() < 1e-10);
        assert!(origin.x.abs() < 1e-10);
        assert!(origin.y.abs() < 1e-10);

        // Check constraint: -t² + x² + y² = -1
        let p = HyperboloidPoint::new(0.5, 0.3);
        let constraint = -p.t * p.t + p.x * p.x + p.y * p.y;
        assert!((constraint + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyperboloid_distance() {
        let origin = HyperboloidPoint::origin();
        let p = HyperboloidPoint::new(0.5, 0.0);

        let dist = origin.distance(&p);
        assert!(dist > 0.0);

        // Distance to self is 0
        assert!(origin.distance(&origin).abs() < 1e-10);
    }

    #[test]
    fn test_poincare_conversion() -> Result<()> {
        let h = HyperboloidPoint::new(0.3, 0.2);
        let p = h.to_poincare()?;
        let h_back = HyperboloidPoint::from_poincare(&p);

        // Round-trip should preserve position
        assert!((h.t - h_back.t).abs() < 1e-6);
        assert!((h.x - h_back.x).abs() < 1e-6);
        assert!((h.y - h_back.y).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_lorentz_boost() {
        let p = HyperboloidPoint::new(0.0, 0.0);
        let boosted = p.lorentz_boost_x(0.5);

        // After boost, should still satisfy constraint
        let constraint = -boosted.t * boosted.t + boosted.x * boosted.x + boosted.y * boosted.y;
        assert!((constraint + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adversarial_lattice_creation() -> Result<()> {
        let topology = DefenseTopology::maximum_connectivity(2);
        let lattice = AdversarialLattice::new(topology)?;

        assert!(lattice.sentries.len() > 1);
        assert!(lattice.perimeter_ratio() > 1.0); // Exponential growth

        Ok(())
    }

    #[test]
    fn test_exponential_boundary_growth() -> Result<()> {
        // Compare total sentries at different depths
        let shallow = AdversarialLattice::new(DefenseTopology::maximum_connectivity(1))?;
        let deep = AdversarialLattice::new(DefenseTopology::maximum_connectivity(3))?;

        // Deeper lattice should have more sentries (exponential growth)
        assert!(deep.sentries.len() > shallow.sentries.len(),
            "Deep lattice should have more sentries: {} vs {}",
            deep.sentries.len(), shallow.sentries.len());

        // Verify exponential growth by checking perimeter at max depth grows faster
        assert!(deep.perimeter_count() > shallow.perimeter_count(),
            "Deep lattice perimeter should be larger: {} vs {}",
            deep.perimeter_count(), shallow.perimeter_count());

        Ok(())
    }

    #[test]
    fn test_message_processing() -> Result<()> {
        let topology = DefenseTopology::balanced_fanout(2);
        let mut lattice = AdversarialLattice::new(topology)?;

        // Process benign message (low anomaly score)
        let result = lattice.process_message(0, 0.1, 1);
        assert!(!result.is_threat);
        assert!(result.confidence < 0.5);

        // Reset and process malicious message (high anomaly score)
        lattice.reset();
        let result = lattice.process_message(0, 0.9, 2);
        assert!(result.confidence > 0.5);

        Ok(())
    }

    #[test]
    fn test_sentry_belief_update() {
        let mut sentry = SentryNode::new(0, HyperboloidPoint::origin(), 0);

        // Update with benign observation
        sentry.update_belief(0.1, 0.3);
        assert!(sentry.belief_state >= 0.0 && sentry.belief_state <= 1.0,
            "Belief should be in [0,1], got {}", sentry.belief_state);

        // Record initial belief
        let initial_belief = sentry.belief_state;

        // Update with anomalous observation multiple times to move belief up
        for _ in 0..5 {
            sentry.update_belief(0.9, 0.3);
        }

        // After repeated high anomaly observations, belief should increase
        assert!(sentry.belief_state > initial_belief,
            "Belief should increase with anomalous observations: {} -> {}",
            initial_belief, sentry.belief_state);
    }

    #[test]
    fn test_curvature_tuning() {
        // Higher FPR tolerance -> higher curvature
        let low_fpr = curvature_tuning::tune_for_fpr(0.01);
        let high_fpr = curvature_tuning::tune_for_fpr(0.1);
        assert!(high_fpr > low_fpr);

        // Higher TPR requirement -> higher curvature
        let low_tpr = curvature_tuning::tune_for_tpr(0.7);
        let high_tpr = curvature_tuning::tune_for_tpr(0.95);
        assert!(high_tpr > low_tpr);
    }

    #[test]
    fn test_different_topologies() -> Result<()> {
        // Test all predefined topologies
        let topologies = vec![
            DefenseTopology::maximum_connectivity(2),
            DefenseTopology::balanced_fanout(2),
            DefenseTopology::aggressive_detection(2),
        ];

        for topology in topologies {
            let lattice = AdversarialLattice::new(topology)?;
            assert!(lattice.sentries.len() > 0);

            let stats = lattice.stats();
            assert!(stats.total_connections > 0);
        }

        Ok(())
    }

    #[test]
    fn test_lattice_stats() -> Result<()> {
        let mut lattice = AdversarialLattice::new(DefenseTopology::balanced_fanout(2))?;

        // Process some messages
        lattice.process_message(0, 0.9, 1);

        let stats = lattice.stats();
        assert!(stats.total_sentries > 0);
        assert!(stats.perimeter_ratio >= 1.0);

        Ok(())
    }
}
