//! # Hyperbolic Geometry Module
//!
//! Provides hyperbolic geometric operations in H^11 (Lorentz model) and
//! Poincaré ball model, integrating with tengri-holographic-cortex.
//!
//! ## Mathematical Foundation
//!
//! ### Lorentz Model (Hyperboloid)
//! - Space: H^n = {x ∈ R^{n+1} : ⟨x,x⟩_L = -1, x₀ > 0}
//! - Metric: ds² = -dx₀² + Σdxᵢ²
//! - Inner product: ⟨x,y⟩_L = -x₀y₀ + Σxᵢyᵢ
//!
//! ### Poincaré Ball Model
//! - Space: B^n = {x ∈ R^n : ||x|| < 1}
//! - Metric: ds² = 4(dx²)/(1-||x||²)²
//! - Möbius addition: x ⊕_M y = [(1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y] / D
//!
//! ## Features
//!
//! - H^11 Lorentz point operations
//! - Poincaré ball embeddings
//! - Geodesic distance computation
//! - Exponential and logarithmic maps
//! - Parallel transport
//! - Curvature-aware attention

use std::f64::consts::PI;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

/// Dimension of hyperbolic space (spatial)
pub const HYPERBOLIC_DIM: usize = 11;

/// Dimension of Lorentz space (time + spatial)
pub const LORENTZ_DIM: usize = 12;

/// Default curvature (negative for hyperbolic)
pub const DEFAULT_CURVATURE: f64 = -1.0;

/// Golden ratio
pub const PHI: f64 = 1.618033988749895;

/// Inverse golden ratio
pub const PHI_INV: f64 = 0.618033988749895;

/// Numerical tolerance
const EPSILON: f64 = 1e-10;

/// Maximum norm for Poincaré ball (avoids boundary)
const MAX_NORM: f64 = 1.0 - 1e-5;

// ============================================================================
// Lorentz Point (H^11)
// ============================================================================

/// Point in Lorentz (hyperboloid) model of H^11
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LorentzPoint11 {
    /// Coordinates: [time, x1, x2, ..., x11]
    coords: [f64; LORENTZ_DIM],
}

impl Default for LorentzPoint11 {
    fn default() -> Self {
        Self::origin()
    }
}

impl LorentzPoint11 {
    /// Create origin point (1, 0, 0, ..., 0)
    pub fn origin() -> Self {
        let mut coords = [0.0; LORENTZ_DIM];
        coords[0] = 1.0;
        Self { coords }
    }

    /// Create from slice (validates Lorentz constraint)
    pub fn from_slice(data: &[f64]) -> Option<Self> {
        if data.len() != LORENTZ_DIM {
            return None;
        }
        let mut coords = [0.0; LORENTZ_DIM];
        coords.copy_from_slice(data);
        let point = Self { coords };
        if point.is_valid() {
            Some(point)
        } else {
            None
        }
    }

    /// Create from Poincaré ball coordinates
    pub fn from_poincare(ball: &[f64]) -> Self {
        let norm_sq: f64 = ball.iter().map(|x| x * x).sum();
        let norm_sq = norm_sq.min(MAX_NORM * MAX_NORM);

        let mut coords = [0.0; LORENTZ_DIM];
        coords[0] = (1.0 + norm_sq) / (1.0 - norm_sq);

        let scale = 2.0 / (1.0 - norm_sq);
        for (i, &x) in ball.iter().enumerate().take(HYPERBOLIC_DIM) {
            coords[i + 1] = scale * x;
        }

        Self { coords }
    }

    /// Create from tangent vector at origin (exponential map)
    pub fn from_tangent_at_origin(tangent: &[f64]) -> Self {
        let norm: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm < EPSILON {
            return Self::origin();
        }

        let mut coords = [0.0; LORENTZ_DIM];
        coords[0] = norm.cosh();

        let sinh_norm = norm.sinh();
        for (i, &t) in tangent.iter().enumerate().take(HYPERBOLIC_DIM) {
            coords[i + 1] = sinh_norm * t / norm;
        }

        Self { coords }
    }

    /// Get coordinates as slice
    pub fn coords(&self) -> &[f64; LORENTZ_DIM] {
        &self.coords
    }

    /// Get time component
    pub fn time(&self) -> f64 {
        self.coords[0]
    }

    /// Get spatial components
    pub fn spatial(&self) -> &[f64] {
        &self.coords[1..]
    }

    /// Validate Lorentz constraint: ⟨x,x⟩_L = -1
    pub fn is_valid(&self) -> bool {
        let inner = self.lorentz_norm();
        (inner + 1.0).abs() < 1e-4
    }

    /// Compute Lorentz norm: ⟨x,x⟩_L
    pub fn lorentz_norm(&self) -> f64 {
        -self.coords[0] * self.coords[0]
            + self.coords[1..].iter().map(|x| x * x).sum::<f64>()
    }

    /// Project back onto hyperboloid (numerical stability)
    pub fn project(&mut self) {
        let spatial_norm_sq: f64 = self.coords[1..].iter().map(|x| x * x).sum();
        self.coords[0] = (1.0 + spatial_norm_sq).sqrt();
    }

    /// Convert to Poincaré ball coordinates
    pub fn to_poincare(&self) -> Vec<f64> {
        let denom = 1.0 + self.coords[0];
        self.coords[1..]
            .iter()
            .map(|&x| x / denom)
            .collect()
    }

    /// Lorentz inner product with another point
    pub fn inner(&self, other: &Self) -> f64 {
        lorentz_inner(&self.coords, &other.coords)
    }

    /// Hyperbolic distance to another point
    pub fn distance(&self, other: &Self) -> f64 {
        hyperbolic_distance(&self.coords, &other.coords)
    }

    /// Geodesic interpolation (t ∈ [0, 1])
    pub fn geodesic(&self, other: &Self, t: f64) -> Self {
        let d = self.distance(other);
        if d < EPSILON {
            return self.clone();
        }

        let mut coords = [0.0; LORENTZ_DIM];
        let sinh_d = d.sinh();
        let t1 = ((1.0 - t) * d).sinh() / sinh_d;
        let t2 = (t * d).sinh() / sinh_d;

        for i in 0..LORENTZ_DIM {
            coords[i] = t1 * self.coords[i] + t2 * other.coords[i];
        }

        let mut result = Self { coords };
        result.project();
        result
    }

    /// Logarithmic map (tangent vector at self pointing to other)
    pub fn log_map(&self, other: &Self) -> Vec<f64> {
        let d = self.distance(other);
        if d < EPSILON {
            return vec![0.0; HYPERBOLIC_DIM];
        }

        let inner = self.inner(other);
        let scale = d / (inner * inner - 1.0).sqrt().max(EPSILON);

        self.coords[1..]
            .iter()
            .zip(&other.coords[1..])
            .map(|(&s, &o)| scale * (o + inner * s))
            .collect()
    }

    /// Parallel transport a tangent vector from origin to self
    pub fn parallel_transport_from_origin(&self, tangent: &[f64]) -> Vec<f64> {
        // Simplified parallel transport for origin → self
        let alpha = self.coords[0];
        let mut result = tangent.to_vec();

        // Scale by cosh(d) factor
        let d = stable_acosh(alpha);
        if d > EPSILON {
            let factor = d.sinh() / d;
            for v in &mut result {
                *v *= factor;
            }
        }

        result
    }
}

// ============================================================================
// Poincaré Ball Operations
// ============================================================================

/// Point in Poincaré ball model
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PoincareBallPoint {
    /// Coordinates (must satisfy ||x|| < 1)
    coords: Vec<f64>,
    /// Curvature (typically -1)
    curvature: f64,
}

impl PoincareBallPoint {
    /// Create origin point
    pub fn origin(dim: usize) -> Self {
        Self {
            coords: vec![0.0; dim],
            curvature: DEFAULT_CURVATURE,
        }
    }

    /// Create from coordinates (projects if necessary)
    pub fn new(coords: Vec<f64>, curvature: f64) -> Self {
        let mut point = Self { coords, curvature };
        point.project();
        point
    }

    /// Get coordinates
    pub fn coords(&self) -> &[f64] {
        &self.coords
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    /// Get curvature
    pub fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Compute squared norm
    pub fn norm_sq(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum()
    }

    /// Compute norm
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Project to ensure ||x|| < 1
    pub fn project(&mut self) {
        let norm = self.norm();
        if norm >= MAX_NORM {
            let scale = MAX_NORM / norm;
            for x in &mut self.coords {
                *x *= scale;
            }
        }
    }

    /// Conformal factor at this point
    pub fn conformal_factor(&self) -> f64 {
        2.0 / (1.0 - self.norm_sq() * (-self.curvature))
    }

    /// Möbius addition: self ⊕ other
    pub fn mobius_add(&self, other: &Self) -> Self {
        let c = -self.curvature;
        let result = mobius_add(&self.coords, &other.coords, c);
        Self::new(result, self.curvature)
    }

    /// Möbius scalar multiplication
    pub fn mobius_scalar(&self, r: f64) -> Self {
        let norm = self.norm();
        if norm < EPSILON {
            return self.clone();
        }

        let c = (-self.curvature).sqrt();
        let new_norm = ((c * norm).tanh() * r).tanh() / c;
        let scale = new_norm / norm;

        let coords: Vec<f64> = self.coords.iter().map(|&x| x * scale).collect();
        Self::new(coords, self.curvature)
    }

    /// Distance to another point
    pub fn distance(&self, other: &Self) -> f64 {
        poincare_distance(&self.coords, &other.coords, self.curvature)
    }

    /// Convert to Lorentz model
    pub fn to_lorentz(&self) -> LorentzPoint11 {
        LorentzPoint11::from_poincare(&self.coords)
    }

    /// Geodesic midpoint
    pub fn midpoint(&self, other: &Self) -> Self {
        // Use Lorentz for stable geodesic computation
        let l1 = self.to_lorentz();
        let l2 = other.to_lorentz();
        let mid = l1.geodesic(&l2, 0.5);
        let coords = mid.to_poincare();
        Self::new(coords, self.curvature)
    }
}

// ============================================================================
// Core Functions
// ============================================================================

/// Lorentz inner product: ⟨x,y⟩_L = -x₀y₀ + Σxᵢyᵢ
#[inline]
pub fn lorentz_inner(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    -x[0] * y[0] + x[1..].iter().zip(&y[1..]).map(|(a, b)| a * b).sum::<f64>()
}

/// Hyperbolic distance in Lorentz model
#[inline]
pub fn hyperbolic_distance(x: &[f64], y: &[f64]) -> f64 {
    let inner = -lorentz_inner(x, y);
    stable_acosh(inner.max(1.0))
}

/// Stable acosh computation (avoids NaN for x ≈ 1)
#[inline]
pub fn stable_acosh(x: f64) -> f64 {
    if x < 1.0001 {
        (2.0 * (x - 1.0).max(0.0)).sqrt()
    } else {
        x.acosh()
    }
}

/// Möbius addition in Poincaré ball
pub fn mobius_add(x: &[f64], y: &[f64], c: f64) -> Vec<f64> {
    let xy: f64 = x.iter().zip(y).map(|(a, b)| a * b).sum();
    let x_norm_sq: f64 = x.iter().map(|a| a * a).sum();
    let y_norm_sq: f64 = y.iter().map(|a| a * a).sum();

    let denom = 1.0 + 2.0 * c * xy + c * c * x_norm_sq * y_norm_sq;
    let coef_x = 1.0 + 2.0 * c * xy + c * y_norm_sq;
    let coef_y = 1.0 - c * x_norm_sq;

    x.iter()
        .zip(y)
        .map(|(xi, yi)| (coef_x * xi + coef_y * yi) / denom)
        .collect()
}

/// Poincaré ball distance
pub fn poincare_distance(x: &[f64], y: &[f64], curvature: f64) -> f64 {
    let c = -curvature;
    let diff: Vec<f64> = x.iter().zip(y).map(|(a, b)| a - b).collect();
    let diff_norm_sq: f64 = diff.iter().map(|d| d * d).sum();

    let x_norm_sq: f64 = x.iter().map(|a| a * a).sum();
    let y_norm_sq: f64 = y.iter().map(|a| a * a).sum();

    let num = 2.0 * c * diff_norm_sq;
    let denom = (1.0 - c * x_norm_sq) * (1.0 - c * y_norm_sq);

    let ratio = 1.0 + num / denom.max(EPSILON);
    stable_acosh(ratio) / c.sqrt()
}

/// Exponential map at origin (tangent → manifold)
pub fn exp_map_origin(tangent: &[f64], curvature: f64) -> Vec<f64> {
    let c = (-curvature).sqrt();
    let norm: f64 = tangent.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm < EPSILON {
        return tangent.to_vec();
    }

    let factor = (c * norm).tanh() / (c * norm);
    tangent.iter().map(|&t| t * factor).collect()
}

/// Logarithmic map at origin (manifold → tangent)
pub fn log_map_origin(point: &[f64], curvature: f64) -> Vec<f64> {
    let c = (-curvature).sqrt();
    let norm: f64 = point.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm < EPSILON {
        return point.to_vec();
    }

    let factor = (c * norm).atanh() / (c * norm);
    point.iter().map(|&p| p * factor).collect()
}

// ============================================================================
// Hyperbolic Neural Operations
// ============================================================================

/// Hyperbolic centroid (Fréchet mean approximation)
pub fn hyperbolic_centroid(points: &[LorentzPoint11]) -> LorentzPoint11 {
    if points.is_empty() {
        return LorentzPoint11::origin();
    }
    if points.len() == 1 {
        return points[0].clone();
    }

    // Einstein midpoint in Lorentz model
    let n = points.len() as f64;
    let mut sum = [0.0; LORENTZ_DIM];

    for p in points {
        for (i, &c) in p.coords().iter().enumerate() {
            sum[i] += c;
        }
    }

    // Average and project
    for s in &mut sum {
        *s /= n;
    }

    let mut result = LorentzPoint11 { coords: sum };
    result.project();
    result
}

/// Hyperbolic attention weight (curvature-aware)
pub fn hyperbolic_attention(query: &LorentzPoint11, key: &LorentzPoint11, temperature: f64) -> f64 {
    let d = query.distance(key);
    (-d / temperature).exp()
}

/// Batch hyperbolic distances (optimized)
pub fn batch_distances(anchor: &LorentzPoint11, points: &[LorentzPoint11]) -> Vec<f64> {
    points.iter().map(|p| anchor.distance(p)).collect()
}

// ============================================================================
// Pentagon Topology Integration
// ============================================================================

/// Golden ratio phases for 5-engine pentagon topology
pub const PENTAGON_PHASES: [f64; 5] = [0.0, 72.0, 144.0, 216.0, 288.0];

/// Pentagon vertex in hyperbolic space
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PentagonVertex {
    /// Vertex index (0-4)
    pub index: usize,
    /// Position in H^11
    pub position: LorentzPoint11,
    /// Phase angle (degrees)
    pub phase: f64,
}

impl PentagonVertex {
    /// Create pentagon vertices in hyperbolic space
    pub fn create_pentagon(radius: f64) -> [Self; 5] {
        let angle_rad = 2.0 * PI / 5.0;

        std::array::from_fn(|i| {
            let theta = i as f64 * angle_rad;
            let mut tangent = [0.0; HYPERBOLIC_DIM];
            tangent[0] = radius * theta.cos();
            tangent[1] = radius * theta.sin();

            Self {
                index: i,
                position: LorentzPoint11::from_tangent_at_origin(&tangent),
                phase: PENTAGON_PHASES[i],
            }
        })
    }

    /// Get coupling strength to another vertex (golden ratio based)
    pub fn coupling_to(&self, other: &Self) -> f64 {
        let diff = ((self.index as i32 - other.index as i32).abs()) % 5;
        match diff {
            0 => 0.0,
            1 | 4 => PHI / 2.0,     // Adjacent
            2 | 3 => PHI_INV / 2.0, // Skip-one
            _ => 0.0,
        }
    }
}

/// Pentagon topology for 5-engine pBit system
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PentagonTopology {
    /// Vertices
    pub vertices: [PentagonVertex; 5],
    /// Radius in hyperbolic space
    pub radius: f64,
}

impl PentagonTopology {
    /// Create new pentagon topology
    pub fn new(radius: f64) -> Self {
        Self {
            vertices: PentagonVertex::create_pentagon(radius),
            radius,
        }
    }

    /// Get coupling matrix
    pub fn coupling_matrix(&self) -> [[f64; 5]; 5] {
        let mut matrix = [[0.0; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                matrix[i][j] = self.vertices[i].coupling_to(&self.vertices[j]);
            }
        }
        matrix
    }

    /// Compute total coupling energy
    pub fn coupling_energy(&self, states: &[f64; 5]) -> f64 {
        let matrix = self.coupling_matrix();
        let mut energy = 0.0;
        for i in 0..5 {
            for j in (i + 1)..5 {
                energy -= matrix[i][j] * states[i] * states[j];
            }
        }
        energy
    }

    /// Compute phase coherence (Kuramoto order parameter)
    pub fn phase_coherence(&self, phases: &[f64; 5]) -> f64 {
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for &phase in phases {
            sum_cos += phase.cos();
            sum_sin += phase.sin();
        }
        (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / 5.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lorentz_origin() {
        let origin = LorentzPoint11::origin();
        assert!(origin.is_valid());
        assert_eq!(origin.time(), 1.0);
    }

    #[test]
    fn test_lorentz_distance_symmetry() {
        let p1 = LorentzPoint11::from_tangent_at_origin(&[0.5, 0.0, 0.0]);
        let p2 = LorentzPoint11::from_tangent_at_origin(&[0.0, 0.5, 0.0]);

        assert!((p1.distance(&p2) - p2.distance(&p1)).abs() < EPSILON);
    }

    #[test]
    fn test_lorentz_triangle_inequality() {
        let p1 = LorentzPoint11::origin();
        let p2 = LorentzPoint11::from_tangent_at_origin(&[1.0, 0.0, 0.0]);
        let p3 = LorentzPoint11::from_tangent_at_origin(&[0.0, 1.0, 0.0]);

        let d12 = p1.distance(&p2);
        let d23 = p2.distance(&p3);
        let d13 = p1.distance(&p3);

        assert!(d13 <= d12 + d23 + EPSILON);
    }

    #[test]
    fn test_poincare_conversion() {
        let ball_coords = vec![0.3, 0.2, 0.0, 0.0, 0.0];
        let lorentz = LorentzPoint11::from_poincare(&ball_coords);
        assert!(lorentz.is_valid());

        let back = lorentz.to_poincare();
        for (a, b) in ball_coords.iter().zip(&back) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mobius_add_identity() {
        let x = vec![0.3, 0.2, 0.1];
        let zero = vec![0.0, 0.0, 0.0];
        let result = mobius_add(&x, &zero, 1.0);

        for (a, b) in x.iter().zip(&result) {
            assert!((a - b).abs() < EPSILON);
        }
    }

    #[test]
    fn test_geodesic_endpoints() {
        let p1 = LorentzPoint11::from_tangent_at_origin(&[0.5, 0.0, 0.0]);
        let p2 = LorentzPoint11::from_tangent_at_origin(&[0.0, 0.5, 0.0]);

        let start = p1.geodesic(&p2, 0.0);
        let end = p1.geodesic(&p2, 1.0);

        assert!(p1.distance(&start) < EPSILON);
        assert!(p2.distance(&end) < EPSILON);
    }

    #[test]
    fn test_pentagon_coupling() {
        let topology = PentagonTopology::new(1.0);
        let matrix = topology.coupling_matrix();

        // Adjacent coupling should be PHI/2
        assert!((matrix[0][1] - PHI / 2.0).abs() < EPSILON);

        // Skip-one coupling should be PHI_INV/2
        assert!((matrix[0][2] - PHI_INV / 2.0).abs() < EPSILON);

        // Self-coupling should be 0
        assert_eq!(matrix[0][0], 0.0);
    }

    #[test]
    fn test_phase_coherence() {
        let topology = PentagonTopology::new(1.0);

        // All same phase → coherence = 1
        let same = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((topology.phase_coherence(&same) - 1.0).abs() < EPSILON);

        // Uniform distribution → low coherence
        let uniform: [f64; 5] = std::array::from_fn(|i| 2.0 * PI * i as f64 / 5.0);
        assert!(topology.phase_coherence(&uniform) < 0.1);
    }
}
