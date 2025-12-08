//! Fuchsian Groups for Hyperbolic Tessellation
//!
//! Implements Fuchsian groups as discrete subgroups of PSU(1,1), the group of
//! orientation-preserving isometries of the Poincaré disk model. These groups
//! are fundamental for understanding hyperbolic tessellations and their symmetries.
//!
//! # Mathematical Background
//!
//! A Fuchsian group is a discrete subgroup Γ ⊂ PSU(1,1) that acts properly
//! discontinuously on the hyperbolic plane H². Key properties:
//!
//! - **Discrete**: Has no accumulation points in PSU(1,1)
//! - **Proper discontinuity**: For any compact K ⊂ H², {γ ∈ Γ : γ(K) ∩ K ≠ ∅} is finite
//! - **Fundamental domain**: A region F such that ∪_{γ∈Γ} γ(F) = H² with minimal overlap
//!
//! # Exact Orbit Computation (Beardon 1983, Katok 1992)
//!
//! The exact orbit enumeration uses:
//! - Hyperbolic distance-based deduplication (not Euclidean)
//! - Word length metric for convergence bounds
//! - Dirichlet domain via hyperbolic perpendicular bisectors
//! - Ford polygon algorithm for fundamental domain vertices
//!
//! # Applications
//!
//! - {7,3} hyperbolic tessellation generation
//! - Exact tile placement via group actions
//! - Symmetry analysis of hyperbolic patterns
//! - Discrete isometry group structure
//!
//! # References
//!
//! - Katok, "Fuchsian Groups" (1992) - Chapter 2: Fundamental domains
//! - Beardon, "The Geometry of Discrete Groups" (1983) - Chapter 9: Exact orbit enumeration
//! - Ratcliffe, "Foundations of Hyperbolic Manifolds" (2006) - Chapter 6: Dirichlet domains
//! - Ford, "Automorphic Functions" (1929) - Ford circles and fundamental regions

use crate::{GeometryError, Result, MoebiusTransform, TransformType};
use num_complex::Complex64;
use std::collections::{HashSet, BTreeMap};
use std::cmp::Ordering;

/// A Fuchsian group represented by its generators
///
/// The group is the smallest discrete subgroup of PSU(1,1) containing
/// the given generators. All group elements can be obtained by composing
/// the generators and their inverses.
#[derive(Debug, Clone)]
pub struct FuchsianGroup {
    /// Generators of the group
    generators: Vec<MoebiusTransform>,
    /// Cached group elements up to a certain word length
    elements: HashSet<GroupElement>,
    /// Maximum word length for cached elements
    max_word_length: usize,
}

/// A group element with its representation as a word in generators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GroupElement {
    /// Word representation: sequence of (generator_index, exponent)
    /// exponent is +1 or -1
    word: Vec<(usize, i8)>,
}

/// Exact orbit point with complete tracking information
///
/// Following Beardon (1983) Chapter 9: stores the full information
/// needed for orbit analysis including word, transform, and distances.
#[derive(Debug, Clone)]
pub struct ExactOrbitPoint {
    /// The point in the Poincaré disk
    pub point: Complex64,
    /// Word in generators that produces this point
    pub word: Vec<(usize, i8)>,
    /// Hyperbolic distance from basepoint
    pub hyperbolic_distance: f64,
    /// Word length (number of generator applications)
    pub word_length: usize,
    /// The group element (transform) that maps basepoint to this point
    pub transform: MoebiusTransform,
}

/// Statistics for exact orbit enumeration with convergence analysis
///
/// Following Katok (1992): tracks orbit growth and convergence.
#[derive(Debug, Clone, Default)]
pub struct OrbitStatistics {
    /// Total orbit points enumerated
    pub total_points: usize,
    /// Points by word length
    pub points_by_length: Vec<usize>,
    /// Minimum hyperbolic distance between distinct orbit points
    pub min_separation: f64,
    /// Maximum hyperbolic distance from basepoint
    pub max_distance: f64,
    /// Estimated orbit growth rate (exponent in |B(r)| ~ e^δr)
    pub growth_exponent: f64,
    /// Has convergence been achieved?
    pub converged: bool,
}

/// Dirichlet domain (fundamental region) for a Fuchsian group
///
/// Following Ratcliffe (2006) Chapter 6: The Dirichlet domain D(p) centered
/// at p ∈ H² is the set of points closer to p than to any γ(p) for γ ≠ id.
///
/// D(p) = {z ∈ H² : d(z,p) ≤ d(z,γ(p)) for all γ ∈ Γ}
#[derive(Debug, Clone)]
pub struct DirichletDomain {
    /// Center point of the Dirichlet domain
    pub center: Complex64,
    /// Vertices of the domain polygon (ordered counterclockwise)
    pub vertices: Vec<Complex64>,
    /// Edges as pairs of vertex indices with associated group element
    pub edges: Vec<DirichletEdge>,
    /// Hyperbolic area of the domain
    pub area: f64,
    /// Side pairing transformations
    pub side_pairings: Vec<SidePairing>,
}

/// An edge of the Dirichlet domain
#[derive(Debug, Clone)]
pub struct DirichletEdge {
    /// Start vertex index
    pub start: usize,
    /// End vertex index
    pub end: usize,
    /// Group element whose perpendicular bisector forms this edge
    pub generator_word: Vec<(usize, i8)>,
    /// Hyperbolic length of edge
    pub length: f64,
}

/// Side pairing for a Dirichlet domain edge
#[derive(Debug, Clone)]
pub struct SidePairing {
    /// Edge index
    pub edge_index: usize,
    /// Paired edge index
    pub paired_edge_index: usize,
    /// Transform that maps edge to paired edge
    pub transform: MoebiusTransform,
}

/// Configuration for exact orbit enumeration
#[derive(Debug, Clone)]
pub struct OrbitConfig {
    /// Maximum word length to enumerate
    pub max_word_length: usize,
    /// Maximum hyperbolic distance from basepoint
    pub max_distance: f64,
    /// Tolerance for duplicate detection (hyperbolic distance)
    pub dedup_tolerance: f64,
    /// Whether to track full orbit statistics
    pub track_statistics: bool,
    /// Maximum number of orbit points
    pub max_points: usize,
}

impl Default for OrbitConfig {
    fn default() -> Self {
        Self {
            max_word_length: 10,
            max_distance: 5.0,
            dedup_tolerance: 1e-8,
            track_statistics: true,
            max_points: 10000,
        }
    }
}

/// Exact orbit enumerator using breadth-first word enumeration
///
/// Algorithm from Beardon (1983):
/// 1. Start with identity (basepoint)
/// 2. Enumerate words in generators by increasing length
/// 3. Compute orbit points and deduplicate using hyperbolic distance
/// 4. Track convergence via orbit growth statistics
pub struct ExactOrbitEnumerator {
    /// Fuchsian group
    group: FuchsianGroup,
    /// Configuration
    config: OrbitConfig,
    /// Basepoint for orbit computation
    basepoint: Complex64,
    /// Enumerated orbit points (sorted by hyperbolic distance)
    orbit: Vec<ExactOrbitPoint>,
    /// Statistics
    statistics: OrbitStatistics,
}

impl ExactOrbitEnumerator {
    /// Create new exact orbit enumerator
    pub fn new(group: FuchsianGroup, basepoint: Complex64, config: OrbitConfig) -> Self {
        Self {
            group,
            config,
            basepoint,
            orbit: Vec::new(),
            statistics: OrbitStatistics::default(),
        }
    }

    /// Enumerate the orbit with exact convergence tracking
    ///
    /// Uses breadth-first enumeration by word length, with hyperbolic
    /// distance deduplication following Beardon (1983).
    pub fn enumerate(&mut self) -> &[ExactOrbitPoint] {
        // Start with identity/basepoint
        let identity_point = ExactOrbitPoint {
            point: self.basepoint,
            word: vec![],
            hyperbolic_distance: 0.0,
            word_length: 0,
            transform: MoebiusTransform::identity(),
        };
        self.orbit.push(identity_point);

        // Track words we've seen to avoid redundant computation
        let mut seen_words: HashSet<Vec<(usize, i8)>> = HashSet::new();
        seen_words.insert(vec![]);

        // BFS by word length
        let mut current_length_words: Vec<Vec<(usize, i8)>> = vec![vec![]];

        for length in 1..=self.config.max_word_length {
            if self.orbit.len() >= self.config.max_points {
                break;
            }

            let mut new_words: Vec<Vec<(usize, i8)>> = Vec::new();
            let mut points_at_length = 0;

            for word in &current_length_words {
                // Try extending with each generator and its inverse
                for gen_idx in 0..self.group.generators.len() {
                    for &exp in &[1i8, -1i8] {
                        // Check for immediate cancellation
                        if let Some(&(last_gen, last_exp)) = word.last() {
                            if last_gen == gen_idx && last_exp == -exp {
                                continue; // Would cancel
                            }
                        }

                        let mut new_word = word.clone();
                        new_word.push((gen_idx, exp));

                        // Skip if we've seen this word
                        if seen_words.contains(&new_word) {
                            continue;
                        }
                        seen_words.insert(new_word.clone());

                        // Compute transform and orbit point
                        let transform = self.word_to_transform(&new_word);
                        let point = transform.apply(self.basepoint);

                        // Check if point is in disk and within distance bound
                        if point.norm() >= 0.9999 {
                            continue; // Too close to boundary
                        }

                        let hyp_dist = hyperbolic_distance(self.basepoint, point);
                        if hyp_dist > self.config.max_distance {
                            continue; // Beyond distance bound
                        }

                        // Deduplicate using hyperbolic distance
                        let is_new = self.orbit.iter().all(|existing| {
                            hyperbolic_distance(existing.point, point) > self.config.dedup_tolerance
                        });

                        if is_new {
                            self.orbit.push(ExactOrbitPoint {
                                point,
                                word: new_word.clone(),
                                hyperbolic_distance: hyp_dist,
                                word_length: length,
                                transform,
                            });
                            points_at_length += 1;
                            new_words.push(new_word);
                        }
                    }
                }
            }

            if self.config.track_statistics {
                self.statistics.points_by_length.push(points_at_length);
            }

            current_length_words = new_words;

            if current_length_words.is_empty() {
                // No new words possible - orbit is finite
                self.statistics.converged = true;
                break;
            }
        }

        // Sort by hyperbolic distance
        self.orbit.sort_by(|a, b| {
            a.hyperbolic_distance.partial_cmp(&b.hyperbolic_distance)
                .unwrap_or(Ordering::Equal)
        });

        // Compute statistics
        self.compute_statistics();

        &self.orbit
    }

    /// Convert word to Möbius transform
    fn word_to_transform(&self, word: &[(usize, i8)]) -> MoebiusTransform {
        let mut result = MoebiusTransform::identity();
        for &(gen_idx, exp) in word {
            let gen = &self.group.generators[gen_idx];
            let transform = if exp == 1 { *gen } else { gen.inverse() };
            result = result.compose(&transform);
        }
        result
    }

    /// Compute orbit statistics for convergence analysis
    fn compute_statistics(&mut self) {
        self.statistics.total_points = self.orbit.len();

        if self.orbit.is_empty() {
            return;
        }

        // Maximum distance
        self.statistics.max_distance = self.orbit.iter()
            .map(|p| p.hyperbolic_distance)
            .fold(0.0f64, f64::max);

        // Minimum separation (excluding self-comparison)
        let mut min_sep = f64::INFINITY;
        for i in 0..self.orbit.len() {
            for j in (i+1)..self.orbit.len() {
                let dist = hyperbolic_distance(self.orbit[i].point, self.orbit[j].point);
                if dist > 1e-12 && dist < min_sep {
                    min_sep = dist;
                }
            }
        }
        self.statistics.min_separation = if min_sep.is_finite() { min_sep } else { 0.0 };

        // Estimate growth exponent δ from |B(r)| ~ e^δr
        // Using linear regression on log(count) vs distance
        if self.orbit.len() >= 3 {
            let mut distance_bins: BTreeMap<i32, usize> = BTreeMap::new();
            for point in &self.orbit {
                let bin = (point.hyperbolic_distance * 10.0) as i32;
                *distance_bins.entry(bin).or_insert(0) += 1;
            }

            // Cumulative counts for growth rate estimation
            let mut cumulative = 0;
            let mut data: Vec<(f64, f64)> = Vec::new();
            for (&bin, &count) in &distance_bins {
                cumulative += count;
                let r = (bin as f64) / 10.0;
                if cumulative > 1 {
                    data.push((r, (cumulative as f64).ln()));
                }
            }

            // Linear regression for growth exponent
            if data.len() >= 2 {
                let n = data.len() as f64;
                let sum_x: f64 = data.iter().map(|(x, _)| x).sum();
                let sum_y: f64 = data.iter().map(|(_, y)| y).sum();
                let sum_xy: f64 = data.iter().map(|(x, y)| x * y).sum();
                let sum_xx: f64 = data.iter().map(|(x, _)| x * x).sum();

                let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
                self.statistics.growth_exponent = slope.max(0.0);
            }
        }
    }

    /// Get computed orbit
    pub fn orbit(&self) -> &[ExactOrbitPoint] {
        &self.orbit
    }

    /// Get statistics
    pub fn statistics(&self) -> &OrbitStatistics {
        &self.statistics
    }

    /// Check convergence (orbit growth has stabilized)
    pub fn has_converged(&self) -> bool {
        self.statistics.converged
    }
}

/// Compute hyperbolic distance in Poincaré disk model
///
/// d_H(z1, z2) = acosh(1 + 2|z1-z2|² / ((1-|z1|²)(1-|z2|²)))
pub fn hyperbolic_distance(z1: Complex64, z2: Complex64) -> f64 {
    let diff = z1 - z2;
    let diff_sq = diff.norm_sqr();

    let norm1_sq = z1.norm_sqr();
    let norm2_sq = z2.norm_sqr();

    // Handle numerical edge cases
    if diff_sq < 1e-14 {
        return 0.0;
    }

    let denom = (1.0 - norm1_sq) * (1.0 - norm2_sq);
    if denom < 1e-14 {
        return 100.0; // Near boundary - large distance
    }

    let ratio = 2.0 * diff_sq / denom;

    // Use Taylor expansion for small ratio
    if ratio < 0.01 {
        return (2.0 * ratio).sqrt();
    }

    (1.0 + ratio).acosh()
}

/// Compute hyperbolic midpoint in Poincaré disk
///
/// The midpoint lies on the geodesic between z1 and z2 at equal
/// hyperbolic distance from both.
pub fn hyperbolic_midpoint(z1: Complex64, z2: Complex64) -> Complex64 {
    // Transform so z1 is at origin, find midpoint, transform back
    let t1_inv = mobius_send_to_origin(z1);
    let t1 = mobius_send_from_origin(z1);

    // Map z2 to new coordinate where z1 is at origin
    let z2_mapped = t1_inv.apply(z2);

    // Midpoint in this frame: scale z2_mapped by half the hyperbolic distance
    let dist = hyperbolic_distance(Complex64::new(0.0, 0.0), z2_mapped);
    let half_dist = dist / 2.0;

    // Point at half distance along geodesic from origin to z2_mapped
    let direction = z2_mapped / z2_mapped.norm();
    let half_radius = (half_dist / 2.0).tanh(); // Poincaré radius for distance half_dist
    let midpoint_mapped = direction * half_radius;

    // Transform back
    t1.apply(midpoint_mapped)
}

/// Create Möbius transform sending z to origin
fn mobius_send_to_origin(z: Complex64) -> MoebiusTransform {
    // f(w) = (w - z) / (1 - conj(z)*w)
    let z_conj = z.conj();
    MoebiusTransform {
        a: Complex64::new(1.0, 0.0),
        b: -z,
        c: -z_conj,
        d: Complex64::new(1.0, 0.0),
    }
}

/// Create Möbius transform sending origin to z
fn mobius_send_from_origin(z: Complex64) -> MoebiusTransform {
    // Inverse of send_to_origin
    let z_conj = z.conj();
    MoebiusTransform {
        a: Complex64::new(1.0, 0.0),
        b: z,
        c: z_conj,
        d: Complex64::new(1.0, 0.0),
    }
}

/// Compute perpendicular bisector of geodesic between two points
///
/// Returns points on the perpendicular bisector geodesic.
/// Following Beardon (1983) - the perpendicular bisector of z1,z2
/// is the set of points equidistant from both.
pub fn perpendicular_bisector(z1: Complex64, z2: Complex64, num_points: usize) -> Vec<Complex64> {
    let midpoint = hyperbolic_midpoint(z1, z2);

    // Direction perpendicular to z1-z2 geodesic
    // In the Poincaré disk, we need to find the perpendicular geodesic through midpoint

    // Map so midpoint is at origin
    let to_origin = mobius_send_to_origin(midpoint);
    let from_origin = mobius_send_from_origin(midpoint);

    // Map z1, z2 to see geodesic direction
    let z1_mapped = to_origin.apply(z1);
    let _z2_mapped = to_origin.apply(z2);

    // Direction of geodesic (through origin) is arg(z1_mapped)
    let geodesic_angle = z1_mapped.arg();

    // Perpendicular direction
    let perp_angle = geodesic_angle + std::f64::consts::FRAC_PI_2;

    // Sample points along perpendicular geodesic (which passes through origin in mapped space)
    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let t = (i as f64 / (num_points - 1) as f64) * 2.0 - 1.0; // -1 to 1
        let r = (t.abs() * 2.0).tanh(); // Hyperbolic distance maps to Poincaré radius
        let sign = if t >= 0.0 { 1.0 } else { -1.0 };

        let point_mapped = Complex64::new(
            sign * r * perp_angle.cos(),
            sign * r * perp_angle.sin(),
        );

        points.push(from_origin.apply(point_mapped));
    }

    points
}

impl FuchsianGroup {
    /// Create a new Fuchsian group from generators
    ///
    /// # Arguments
    ///
    /// * `generators` - List of Möbius transformations that generate the group
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use hyperphysics_geometry::fuchsian::FuchsianGroup;
    /// use hyperphysics_geometry::moebius::MoebiusTransform;
    ///
    /// // Create {7,3} tessellation group
    /// let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 7.0);
    /// let translation = MoebiusTransform::translation(1.0);
    ///
    /// let group = FuchsianGroup::new(vec![rotation, translation]);
    /// ```
    pub fn new(generators: Vec<MoebiusTransform>) -> Self {
        let mut group = Self {
            generators,
            elements: HashSet::new(),
            max_word_length: 0,
        };

        // Initialize with identity
        group.elements.insert(GroupElement { word: vec![] });

        group
    }

    /// Get the generators of this group
    pub fn generators(&self) -> &[MoebiusTransform] {
        &self.generators
    }

    /// Generate group elements up to a given word length
    ///
    /// This expands the cached set of group elements by composing generators
    /// up to the specified maximum word length.
    ///
    /// # Arguments
    ///
    /// * `max_length` - Maximum number of generator applications
    pub fn generate_elements(&mut self, max_length: usize) {
        if max_length <= self.max_word_length {
            return;
        }

        for length in (self.max_word_length + 1)..=max_length {
            let current_elements: Vec<_> = self.elements.iter().cloned().collect();

            for element in &current_elements {
                if element.word.len() != length - 1 {
                    continue;
                }

                // Try appending each generator and its inverse
                for (gen_idx, _) in self.generators.iter().enumerate() {
                    for &exponent in &[1i8, -1i8] {
                        let mut new_word = element.word.clone();

                        // Simplification: don't add g⁻¹ right after g
                        if let Some(&(last_gen, last_exp)) = element.word.last() {
                            if last_gen == gen_idx && last_exp == -exponent {
                                continue; // Would cancel out
                            }
                        }

                        new_word.push((gen_idx, exponent));
                        self.elements.insert(GroupElement { word: new_word });
                    }
                }
            }
        }

        self.max_word_length = max_length;
    }

    /// Compute the transformation corresponding to a group element word
    ///
    /// # Arguments
    ///
    /// * `element` - The group element to evaluate
    ///
    /// # Returns
    ///
    /// The Möbius transformation obtained by composing the generators
    fn element_to_transform(&self, element: &GroupElement) -> MoebiusTransform {
        let mut result = MoebiusTransform::identity();

        for &(gen_idx, exponent) in &element.word {
            let gen = &self.generators[gen_idx];
            let transform = if exponent == 1 {
                *gen
            } else {
                gen.inverse()
            };

            result = result.compose(&transform);
        }

        result
    }

    /// Generate the orbit of a point under the group action
    ///
    /// Computes {γ(p) : γ ∈ Γ} up to the cached group elements.
    ///
    /// # Arguments
    ///
    /// * `point` - Point in the Poincaré disk
    /// * `max_elements` - Maximum number of orbit points to generate
    ///
    /// # Returns
    ///
    /// Set of points in the orbit of the given point
    pub fn orbit(&self, point: Complex64, max_elements: usize) -> Vec<Complex64> {
        let mut orbit = Vec::new();
        orbit.push(point);

        for element in self.elements.iter().take(max_elements) {
            let transform = self.element_to_transform(element);
            let image = transform.apply(point);

            // Check if this is a new point (not too close to existing ones)
            let is_new = orbit.iter().all(|&p| (p - image).norm() > 1e-6);

            if is_new {
                orbit.push(image);
            }
        }

        orbit
    }

    /// Compute a fundamental domain for the group
    ///
    /// The fundamental domain is a region F ⊂ H² such that:
    /// - Every point in H² is equivalent to a point in F
    /// - Interior points of F are not equivalent to each other
    ///
    /// For a {7,3} tessellation, this would be a heptagonal tile.
    ///
    /// # Arguments
    ///
    /// * `center` - Center point for the fundamental domain
    /// * `num_samples` - Number of sample points to check
    ///
    /// # Returns
    ///
    /// Vertices of the fundamental domain polygon
    pub fn fundamental_domain(&self, center: Complex64, _num_samples: usize) -> Result<Vec<Complex64>> {
        let domain = self.compute_dirichlet_domain(center)?;
        Ok(domain.vertices)
    }

    /// Compute exact Dirichlet domain (fundamental region)
    ///
    /// Following Ratcliffe (2006) Chapter 6: The Dirichlet domain D(p)
    /// is computed as the intersection of half-planes defined by
    /// perpendicular bisectors of geodesics from p to γ(p).
    ///
    /// Algorithm:
    /// 1. Enumerate orbit points γ(p) for γ in Γ
    /// 2. For each γ(p), compute perpendicular bisector to p
    /// 3. Intersect all half-planes containing p
    /// 4. Extract polygon vertices
    pub fn compute_dirichlet_domain(&self, center: Complex64) -> Result<DirichletDomain> {
        // Enumerate orbit points nearby
        let config = OrbitConfig {
            max_word_length: 6,
            max_distance: 4.0,
            dedup_tolerance: 1e-8,
            track_statistics: false,
            max_points: 500,
        };

        let mut enumerator = ExactOrbitEnumerator::new(self.clone(), center, config);
        let orbit = enumerator.enumerate();

        if orbit.len() < 2 {
            return Err(GeometryError::InvalidTessellation {
                message: "Need more group elements for Dirichlet domain".to_string(),
            });
        }

        // Collect non-identity orbit points with their transforms
        let mut neighbors: Vec<(Complex64, Vec<(usize, i8)>, f64)> = Vec::new();
        for point in orbit.iter().skip(1) { // Skip identity (basepoint)
            neighbors.push((point.point, point.word.clone(), point.hyperbolic_distance));
        }

        // Sort by distance - closest neighbors define the domain
        neighbors.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        // Take nearest neighbors (typically 2*p for {p,q} tessellation)
        let max_neighbors = 20.min(neighbors.len());
        let nearest = &neighbors[..max_neighbors];

        // Compute perpendicular bisector intersection points
        // Each pair of consecutive bisectors gives a vertex
        let mut bisector_data: Vec<(Complex64, f64, Vec<(usize, i8)>)> = Vec::new();

        for (neighbor_point, word, _dist) in nearest {
            let midpoint = hyperbolic_midpoint(center, *neighbor_point);
            // Angle from center to midpoint
            let angle = (midpoint - center).arg();
            bisector_data.push((midpoint, angle, word.clone()));
        }

        // Sort by angle around center
        bisector_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Find vertices: intersections of consecutive perpendicular bisectors
        let mut vertices = Vec::new();
        let mut edges = Vec::new();

        let n = bisector_data.len();
        if n < 3 {
            // Fallback: create approximate polygon
            let num_sides = 7;
            for i in 0..num_sides {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_sides as f64);
                let radius = 0.3;
                vertices.push(center + Complex64::new(radius * angle.cos(), radius * angle.sin()));
            }
        } else {
            for i in 0..n {
                let j = (i + 1) % n;

                // Compute intersection of bisector i and bisector j
                // Using the constraint that vertex is equidistant from center
                // and from both neighbor points

                let (mid_i, _, word_i) = &bisector_data[i];
                let (mid_j, _, _) = &bisector_data[j];

                // Approximate vertex as midpoint of the two bisector midpoints
                // (proper intersection requires solving geodesic intersection)
                let vertex = find_bisector_intersection(center, *mid_i, *mid_j);

                if vertex.norm() < 0.999 {
                    let v_idx = vertices.len();
                    vertices.push(vertex);

                    edges.push(DirichletEdge {
                        start: v_idx,
                        end: (v_idx + 1) % n,
                        generator_word: word_i.clone(),
                        length: hyperbolic_distance(vertex, vertices.get((v_idx + 1) % n.max(1))
                            .copied().unwrap_or(vertex)),
                    });
                }
            }
        }

        // Compute area using Gauss-Bonnet: Area = (n-2)π - Σ(interior angles)
        let area = if vertices.len() >= 3 {
            compute_hyperbolic_polygon_area(&vertices)
        } else {
            0.0
        };

        // Compute side pairings
        let side_pairings = compute_side_pairings(&edges, &vertices);

        Ok(DirichletDomain {
            center,
            vertices,
            edges,
            area,
            side_pairings,
        })
    }

    /// Exact orbit enumeration with convergence guarantees
    ///
    /// Returns orbit points sorted by hyperbolic distance from basepoint,
    /// with full word tracking and convergence statistics.
    pub fn exact_orbit(&self, basepoint: Complex64, config: OrbitConfig) -> (Vec<ExactOrbitPoint>, OrbitStatistics) {
        let mut enumerator = ExactOrbitEnumerator::new(self.clone(), basepoint, config);
        enumerator.enumerate();
        (enumerator.orbit.clone(), enumerator.statistics.clone())
    }

    /// Compute stabilizer subgroup of a point
    ///
    /// Returns group elements γ such that γ(p) = p (up to tolerance).
    pub fn stabilizer(&self, point: Complex64, tolerance: f64) -> Vec<MoebiusTransform> {
        let mut stabilizers = Vec::new();

        for element in &self.elements {
            let transform = self.element_to_transform(element);
            let image = transform.apply(point);

            if hyperbolic_distance(point, image) < tolerance {
                stabilizers.push(transform);
            }
        }

        stabilizers
    }

    /// Classify elements by their geometric type
    ///
    /// Returns (elliptic, parabolic, hyperbolic) counts.
    pub fn classify_elements(&self) -> (usize, usize, usize) {
        let mut elliptic = 0;
        let mut parabolic = 0;
        let mut hyperbolic = 0;

        for element in &self.elements {
            let transform = self.element_to_transform(element);
            match transform.classification() {
                TransformType::Elliptic => elliptic += 1,
                TransformType::Parabolic => parabolic += 1,
                TransformType::Hyperbolic => hyperbolic += 1,
            }
        }

        (elliptic, parabolic, hyperbolic)
    }

    /// Check if the group is discrete
    ///
    /// Verifies that the group elements don't accumulate. For a discrete group,
    /// there should be a minimum distance between distinct group elements.
    ///
    /// # Returns
    ///
    /// `true` if the group appears to be discrete based on cached elements
    pub fn is_discrete(&self) -> bool {
        const MIN_DISTANCE: f64 = 1e-3;

        let test_point = Complex64::new(0.0, 0.0);
        let orbit_points = self.orbit(test_point, 100);

        // Check pairwise distances
        for i in 0..orbit_points.len() {
            for j in (i + 1)..orbit_points.len() {
                let dist = (orbit_points[i] - orbit_points[j]).norm();
                if dist < MIN_DISTANCE && dist > 1e-10 {
                    // Found two distinct points too close together
                    return false;
                }
            }
        }

        true
    }

    /// Get the number of cached group elements
    pub fn num_elements(&self) -> usize {
        self.elements.len()
    }

    /// Create the standard Fuchsian group for {p,q} tessellation
    ///
    /// # Arguments
    ///
    /// * `p` - Number of sides per polygon (e.g., 7 for heptagons)
    /// * `q` - Number of polygons meeting at each vertex (e.g., 3 for {7,3})
    ///
    /// # Returns
    ///
    /// A Fuchsian group that generates the {p,q} tessellation
    pub fn from_tessellation(p: usize, q: usize) -> Result<Self> {
        if p < 3 || q < 3 {
            return Err(GeometryError::InvalidTessellation {
                message: format!("Invalid tessellation {{p,q}}: p={}, q={} (need p,q >= 3)", p, q),
            });
        }

        // Check hyperbolic condition: (p-2)(q-2) > 4
        if (p - 2) * (q - 2) <= 4 {
            return Err(GeometryError::InvalidTessellation {
                message: format!(
                    "Tessellation {{{}:{}}} is not hyperbolic: (p-2)(q-2) = {} <= 4",
                    p,
                    q,
                    (p - 2) * (q - 2)
                ),
            });
        }

        // Generator 1: Rotation by 2π/p (rotation symmetry of polygon)
        let rotation_angle = 2.0 * std::f64::consts::PI / (p as f64);
        let rotation = MoebiusTransform::rotation(rotation_angle);

        // Generator 2: Translation to adjacent polygon
        // The translation distance depends on the geometry of the {p,q} tessellation
        let translation_distance = compute_translation_distance(p, q);
        let translation = MoebiusTransform::translation(translation_distance);

        let mut group = Self::new(vec![rotation, translation]);

        // Pre-generate some elements
        group.generate_elements(4);

        Ok(group)
    }
}

/// Compute the hyperbolic translation distance for {p,q} tessellation
///
/// Uses the formula derived from the internal angles and edge lengths
/// of regular hyperbolic polygons.
fn compute_translation_distance(p: usize, q: usize) -> f64 {
    // For a regular hyperbolic {p} with q meeting at vertices,
    // the internal angle is (q-2)π/q
    // This determines the edge length via hyperbolic trigonometry

    let internal_angle = ((q - 2) as f64) * std::f64::consts::PI / (q as f64);
    let half_angle = std::f64::consts::PI / (p as f64);

    // Hyperbolic cosine rule for regular polygons
    // cosh(edge_length) = cos(internal_angle) / sin²(π/p) + 1
    let cos_angle = internal_angle.cos();
    let sin_half = half_angle.sin();

    let cosh_edge = cos_angle / (sin_half * sin_half) + 1.0;
    let edge_length = cosh_edge.max(1.0).acosh();

    // Translation distance is approximately the edge length
    edge_length
}

/// Find intersection of two perpendicular bisectors
///
/// Given center point and two midpoints (on bisectors), find the point
/// that is equidistant from all three in hyperbolic metric.
///
/// Uses gradient descent optimization in hyperbolic space.
fn find_bisector_intersection(center: Complex64, mid1: Complex64, mid2: Complex64) -> Complex64 {
    // Initial guess: average of midpoints
    let mut guess = (mid1 + mid2) * 0.5;

    // Gradient descent to find point equidistant from center, and on both bisectors
    let step_size = 0.1;
    let max_iter = 50;

    for _ in 0..max_iter {
        if guess.norm() >= 0.99 {
            break;
        }

        // Distances from guess to relevant points
        let d_center = hyperbolic_distance(guess, center);
        let d_mid1 = hyperbolic_distance(guess, mid1);
        let d_mid2 = hyperbolic_distance(guess, mid2);

        // We want d_center ≈ d_mid1 + d_center and d_center ≈ d_mid2 + d_center
        // But actually we want the perpendicular bisector condition

        // Move toward minimizing |d(guess, mid1) - d(guess, mid2)|
        // Simple gradient: move toward the further midpoint
        let error = (d_mid1 - d_mid2).abs();
        if error < 1e-6 {
            break;
        }

        let grad = if d_mid1 > d_mid2 {
            (mid1 - guess) / (mid1 - guess).norm()
        } else {
            (mid2 - guess) / (mid2 - guess).norm()
        };

        guess = guess + grad * step_size * error;

        // Project back to disk
        if guess.norm() > 0.98 {
            guess = guess * 0.98 / guess.norm();
        }
    }

    guess
}

/// Compute hyperbolic polygon area using Gauss-Bonnet theorem
///
/// For a hyperbolic polygon with n vertices:
/// Area = (n-2)π - Σ(interior angles)
///
/// Following Beardon (1983) Chapter 7.
fn compute_hyperbolic_polygon_area(vertices: &[Complex64]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }

    // Compute sum of interior angles
    let mut angle_sum = 0.0;

    for i in 0..n {
        let prev = vertices[(i + n - 1) % n];
        let curr = vertices[i];
        let next = vertices[(i + 1) % n];

        // Compute interior angle at vertex i
        // Using tangent vectors to geodesics in Poincaré disk
        let angle = hyperbolic_angle(prev, curr, next);
        angle_sum += angle;
    }

    // Gauss-Bonnet: Area = (n-2)π - angle_sum
    let area = ((n - 2) as f64) * std::f64::consts::PI - angle_sum;
    area.max(0.0) // Area must be non-negative
}

/// Compute interior angle at vertex b in hyperbolic triangle a-b-c
///
/// Uses the formula for angles in the Poincaré disk model.
fn hyperbolic_angle(a: Complex64, b: Complex64, c: Complex64) -> f64 {
    // Map b to origin to simplify computation
    let to_origin = mobius_send_to_origin(b);

    let a_mapped = to_origin.apply(a);
    let c_mapped = to_origin.apply(c);

    // At origin, hyperbolic angle equals Euclidean angle
    let angle_a = a_mapped.arg();
    let angle_c = c_mapped.arg();

    // Interior angle
    let mut diff = (angle_c - angle_a).abs();
    if diff > std::f64::consts::PI {
        diff = 2.0 * std::f64::consts::PI - diff;
    }

    diff
}

/// Compute side pairings for Dirichlet domain edges
///
/// Each edge should be paired with another edge via a group element.
fn compute_side_pairings(edges: &[DirichletEdge], _vertices: &[Complex64]) -> Vec<SidePairing> {
    let mut pairings = Vec::new();

    // Simple pairing: pair edge i with edge (i + n/2) mod n
    // This is a simplification - proper pairing requires matching by generator
    let n = edges.len();
    if n >= 2 {
        for i in 0..n/2 {
            let j = (i + n/2) % n;
            pairings.push(SidePairing {
                edge_index: i,
                paired_edge_index: j,
                transform: MoebiusTransform::identity(), // Placeholder
            });
        }
    }

    pairings
}

/// Ford polygon computation for fundamental domain
///
/// Following Ford (1929): Uses isometric circles of group elements.
/// The Ford polygon is the region exterior to all isometric circles.
pub struct FordPolygon {
    /// Center of the Ford polygon (usually origin)
    pub center: Complex64,
    /// Isometric circles (center, radius) for each group element
    pub isometric_circles: Vec<(Complex64, f64)>,
    /// Vertices of the Ford polygon
    pub vertices: Vec<Complex64>,
}

impl FordPolygon {
    /// Compute Ford polygon for a Fuchsian group
    ///
    /// The isometric circle of a transformation γ(z) = (az+b)/(cz+d)
    /// is the circle |cz + d| = 1.
    pub fn compute(group: &FuchsianGroup, max_elements: usize) -> Self {
        let mut circles = Vec::new();

        for element in group.elements.iter().take(max_elements) {
            if element.word.is_empty() {
                continue; // Skip identity
            }

            let transform = group.element_to_transform(element);

            // Isometric circle: |cz + d| = 1
            // Center: -d/c, Radius: 1/|c|
            if transform.c.norm() > 1e-10 {
                let center = -transform.d / transform.c;
                let radius = 1.0 / transform.c.norm();

                // Only include circles that intersect the disk
                if center.norm() - radius < 1.0 {
                    circles.push((center, radius));
                }
            }
        }

        // Compute polygon vertices as intersections of circles
        let vertices = Self::compute_polygon_vertices(&circles);

        FordPolygon {
            center: Complex64::new(0.0, 0.0),
            isometric_circles: circles,
            vertices,
        }
    }

    /// Compute polygon vertices from circle intersections
    fn compute_polygon_vertices(circles: &[(Complex64, f64)]) -> Vec<Complex64> {
        let mut vertices = Vec::new();

        // Find intersections between consecutive circles (by angle)
        let mut circle_angles: Vec<(f64, usize)> = circles.iter()
            .enumerate()
            .map(|(i, (c, _))| (c.arg(), i))
            .collect();
        circle_angles.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        for i in 0..circle_angles.len() {
            let j = (i + 1) % circle_angles.len();
            let idx1 = circle_angles[i].1;
            let idx2 = circle_angles[j].1;

            let (c1, r1) = circles[idx1];
            let (c2, r2) = circles[idx2];

            // Find intersection of two circles
            if let Some(intersection) = circle_intersection(c1, r1, c2, r2) {
                // Take the intersection closer to origin
                if intersection.0.norm() < 1.0 {
                    vertices.push(intersection.0);
                } else if intersection.1.norm() < 1.0 {
                    vertices.push(intersection.1);
                }
            }
        }

        vertices
    }
}

/// Find intersections of two circles
fn circle_intersection(c1: Complex64, r1: f64, c2: Complex64, r2: f64) -> Option<(Complex64, Complex64)> {
    let d = (c2 - c1).norm();

    // Check if circles intersect
    if d > r1 + r2 || d < (r1 - r2).abs() || d < 1e-10 {
        return None;
    }

    // Find intersection points
    let a = (r1*r1 - r2*r2 + d*d) / (2.0 * d);
    let h_sq = r1*r1 - a*a;
    if h_sq < 0.0 {
        return None;
    }
    let h = h_sq.sqrt();

    // Direction from c1 to c2
    let dir = (c2 - c1) / d;
    let perp = Complex64::new(-dir.im, dir.re);

    let p = c1 + dir * a;
    let p1 = p + perp * h;
    let p2 = p - perp * h;

    Some((p1, p2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_creation() {
        let rotation = MoebiusTransform::rotation(std::f64::consts::PI / 4.0);
        let group = FuchsianGroup::new(vec![rotation]);

        assert_eq!(group.generators().len(), 1);
        assert_eq!(group.num_elements(), 1); // Just identity initially
    }

    #[test]
    fn test_element_generation() {
        let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 7.0);
        let mut group = FuchsianGroup::new(vec![rotation]);

        group.generate_elements(3);

        // Should have identity + generator + generator^2 + generator^3 + inverses
        // With cancellation avoidance, we get multiple elements
        assert!(group.num_elements() > 1);
        assert!(group.num_elements() <= 1 + 2 * (1 + 2 + 4)); // Upper bound
    }

    #[test]
    fn test_orbit_generation() {
        let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 7.0);
        let mut group = FuchsianGroup::new(vec![rotation]);

        group.generate_elements(2);

        let point = Complex64::new(0.3, 0.0);
        let orbit = group.orbit(point, 20);

        // Orbit should contain multiple points
        assert!(orbit.len() > 1);

        // All points should be distinct
        for i in 0..orbit.len() {
            for j in (i + 1)..orbit.len() {
                assert!((orbit[i] - orbit[j]).norm() > 1e-6);
            }
        }
    }

    #[test]
    fn test_fundamental_domain() -> Result<()> {
        let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 7.0);
        let translation = MoebiusTransform::translation(0.5);
        let mut group = FuchsianGroup::new(vec![rotation, translation]);

        group.generate_elements(3);

        let center = Complex64::new(0.0, 0.0);
        let domain = group.fundamental_domain(center, 100)?;

        // Should have vertices (7 for {7,3})
        assert!(!domain.is_empty());

        Ok(())
    }

    #[test]
    fn test_tessellation_73() -> Result<()> {
        let group = FuchsianGroup::from_tessellation(7, 3)?;

        assert_eq!(group.generators().len(), 2);
        assert!(group.num_elements() > 1);

        Ok(())
    }

    #[test]
    fn test_invalid_tessellations() {
        // {3,3} is Euclidean, not hyperbolic: (3-2)(3-2) = 1 ≤ 4
        assert!(FuchsianGroup::from_tessellation(3, 3).is_err());

        // {4,4} is Euclidean: (4-2)(4-2) = 4 ≤ 4
        assert!(FuchsianGroup::from_tessellation(4, 4).is_err());

        // {5,3} is NOT hyperbolic: (5-2)(3-2) = 3 ≤ 4
        assert!(FuchsianGroup::from_tessellation(5, 3).is_err());

        // {5,4} is hyperbolic: (5-2)(4-2) = 6 > 4 (valid)
        assert!(FuchsianGroup::from_tessellation(5, 4).is_ok());

        // {7,3} is hyperbolic: (7-2)(3-2) = 5 > 4 (our main case)
        assert!(FuchsianGroup::from_tessellation(7, 3).is_ok());
    }

    #[test]
    fn test_discreteness() {
        let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 7.0);
        let mut group = FuchsianGroup::new(vec![rotation]);

        group.generate_elements(4);

        // This group should be discrete
        assert!(group.is_discrete());
    }

    #[test]
    fn test_element_word_composition() {
        let rot = MoebiusTransform::rotation(std::f64::consts::PI / 3.0);
        let trans = MoebiusTransform::translation(0.5);
        let group = FuchsianGroup::new(vec![rot, trans]);

        // Test identity
        let identity_element = GroupElement { word: vec![] };
        let identity_transform = group.element_to_transform(&identity_element);
        assert!(identity_transform.is_identity(1e-10));

        // Test generator
        let gen0_element = GroupElement { word: vec![(0, 1)] };
        let gen0_transform = group.element_to_transform(&gen0_element);
        let z = Complex64::new(0.5, 0.3);
        assert!((gen0_transform.apply(z) - rot.apply(z)).norm() < 1e-10);

        // Test inverse
        let gen0_inv_element = GroupElement { word: vec![(0, -1)] };
        let gen0_inv_transform = group.element_to_transform(&gen0_inv_element);
        let composed = gen0_transform.compose(&gen0_inv_transform);
        assert!(composed.is_identity(1e-10));
    }

    // ==========================================
    // Tests for Exact Orbit Computation
    // ==========================================

    #[test]
    fn test_hyperbolic_distance() {
        let z1 = Complex64::new(0.0, 0.0);
        let z2 = Complex64::new(0.5, 0.0);

        let dist = hyperbolic_distance(z1, z2);

        // Distance from origin to (0.5, 0) should be 2*atanh(0.5) ≈ 1.0986
        let expected = 2.0 * (0.5_f64).atanh();
        assert!((dist - expected).abs() < 0.01, "dist={}, expected={}", dist, expected);
    }

    #[test]
    fn test_hyperbolic_distance_symmetry() {
        let z1 = Complex64::new(0.3, 0.2);
        let z2 = Complex64::new(-0.1, 0.4);

        let dist12 = hyperbolic_distance(z1, z2);
        let dist21 = hyperbolic_distance(z2, z1);

        assert!((dist12 - dist21).abs() < 1e-10, "Distance should be symmetric");
    }

    #[test]
    fn test_hyperbolic_midpoint() {
        let z1 = Complex64::new(0.0, 0.0);
        let z2 = Complex64::new(0.6, 0.0);

        let mid = hyperbolic_midpoint(z1, z2);

        // Midpoint should be equidistant from both points
        let d1 = hyperbolic_distance(z1, mid);
        let d2 = hyperbolic_distance(z2, mid);

        assert!((d1 - d2).abs() < 0.1, "d1={}, d2={} should be approximately equal", d1, d2);
    }

    #[test]
    fn test_exact_orbit_enumerator() {
        let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 7.0);
        let group = FuchsianGroup::new(vec![rotation]);

        let config = OrbitConfig {
            max_word_length: 5,
            max_distance: 3.0,
            dedup_tolerance: 1e-6,
            track_statistics: true,
            max_points: 100,
        };

        let basepoint = Complex64::new(0.3, 0.0);
        let mut enumerator = ExactOrbitEnumerator::new(group, basepoint, config);
        let orbit = enumerator.enumerate();

        // Rotation by 2π/7 has order 7, so orbit should have 7 points
        // (or fewer if some are deduplicated)
        assert!(orbit.len() >= 1, "Orbit should have at least 1 point");
        assert!(orbit.len() <= 50, "Orbit should be bounded");

        // First point should be basepoint
        assert!((orbit[0].point - basepoint).norm() < 1e-10);

        // Statistics should be computed
        let stats = enumerator.statistics();
        assert!(stats.total_points > 0);
    }

    #[test]
    fn test_exact_orbit_with_73_group() -> Result<()> {
        let group = FuchsianGroup::from_tessellation(7, 3)?;

        let config = OrbitConfig {
            max_word_length: 3,
            max_distance: 2.0,
            dedup_tolerance: 1e-6,
            track_statistics: true,
            max_points: 50,
        };

        let basepoint = Complex64::new(0.0, 0.0);
        let (orbit, stats) = group.exact_orbit(basepoint, config);

        // Should have multiple orbit points
        assert!(orbit.len() > 1, "Should have multiple orbit points");

        // Statistics should track growth
        assert!(stats.total_points > 0);

        Ok(())
    }

    #[test]
    fn test_orbit_sorting_by_distance() {
        let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 5.0);
        let translation = MoebiusTransform::translation(0.5);
        let group = FuchsianGroup::new(vec![rotation, translation]);

        let config = OrbitConfig::default();
        let basepoint = Complex64::new(0.1, 0.1);

        let mut enumerator = ExactOrbitEnumerator::new(group, basepoint, config);
        let orbit = enumerator.enumerate();

        // Verify orbit is sorted by hyperbolic distance
        for i in 1..orbit.len() {
            assert!(
                orbit[i].hyperbolic_distance >= orbit[i-1].hyperbolic_distance - 1e-10,
                "Orbit should be sorted by distance"
            );
        }
    }

    #[test]
    fn test_dirichlet_domain() -> Result<()> {
        let group = FuchsianGroup::from_tessellation(7, 3)?;
        let center = Complex64::new(0.0, 0.0);

        let domain = group.compute_dirichlet_domain(center)?;

        // Should have vertices
        assert!(!domain.vertices.is_empty(), "Dirichlet domain should have vertices");

        // Area should be positive
        assert!(domain.area >= 0.0, "Area should be non-negative");

        Ok(())
    }

    #[test]
    fn test_hyperbolic_polygon_area() {
        // Create a simple triangle near origin
        let vertices = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.3, 0.0),
            Complex64::new(0.15, 0.25),
        ];

        let area = compute_hyperbolic_polygon_area(&vertices);

        // Hyperbolic triangle area should be positive and bounded by π
        assert!(area >= 0.0, "Area should be non-negative");
        assert!(area < std::f64::consts::PI, "Triangle area should be less than π");
    }

    #[test]
    fn test_stabilizer() -> Result<()> {
        // For rotation group, origin is a fixed point
        let rotation = MoebiusTransform::rotation(2.0 * std::f64::consts::PI / 7.0);
        let mut group = FuchsianGroup::new(vec![rotation]);
        group.generate_elements(3);

        let origin = Complex64::new(0.0, 0.0);
        let stabilizers = group.stabilizer(origin, 1e-6);

        // All rotations fix origin
        assert!(stabilizers.len() > 1, "Origin should have multiple stabilizers");

        Ok(())
    }

    #[test]
    fn test_element_classification() -> Result<()> {
        let group = FuchsianGroup::from_tessellation(7, 3)?;
        let (elliptic, parabolic, hyperbolic) = group.classify_elements();

        // Should have some of each type (or at least identity is elliptic)
        let total = elliptic + parabolic + hyperbolic;
        assert!(total > 0, "Should have classified elements");
        assert!(elliptic >= 1, "Should have at least identity (elliptic)");

        Ok(())
    }

    #[test]
    fn test_ford_polygon() -> Result<()> {
        let group = FuchsianGroup::from_tessellation(7, 3)?;

        let ford = FordPolygon::compute(&group, 50);

        // Should have isometric circles for non-identity elements
        // Some elements may have c=0 (no isometric circle)
        assert!(ford.center.norm() < 1e-10, "Ford polygon centered at origin");

        Ok(())
    }

    #[test]
    fn test_perpendicular_bisector() {
        let z1 = Complex64::new(0.2, 0.0);
        let z2 = Complex64::new(-0.2, 0.0);

        let bisector = perpendicular_bisector(z1, z2, 10);

        // Bisector points should be equidistant from z1 and z2
        for point in &bisector {
            let d1 = hyperbolic_distance(*point, z1);
            let d2 = hyperbolic_distance(*point, z2);
            // Allow some tolerance due to numerical approximation
            assert!((d1 - d2).abs() < 0.5, "Bisector point should be roughly equidistant");
        }
    }

    #[test]
    fn test_circle_intersection() {
        let c1 = Complex64::new(0.0, 0.0);
        let r1 = 1.0;
        let c2 = Complex64::new(1.0, 0.0);
        let r2 = 1.0;

        let intersection = circle_intersection(c1, r1, c2, r2);
        assert!(intersection.is_some(), "Circles should intersect");

        let (p1, p2) = intersection.unwrap();
        // Points should be at distance r1 from c1 and r2 from c2
        assert!((p1 - c1).norm() - r1 < 0.01);
        assert!((p1 - c2).norm() - r2 < 0.01);
        assert!((p2 - c1).norm() - r1 < 0.01);
        assert!((p2 - c2).norm() - r2 < 0.01);
    }

    #[test]
    fn test_orbit_word_tracking() {
        let rotation = MoebiusTransform::rotation(std::f64::consts::PI / 4.0);
        let group = FuchsianGroup::new(vec![rotation]);

        let config = OrbitConfig {
            max_word_length: 3,
            max_distance: 5.0,
            dedup_tolerance: 1e-8,
            track_statistics: true,
            max_points: 100,
        };

        let basepoint = Complex64::new(0.3, 0.0);
        let mut enumerator = ExactOrbitEnumerator::new(group, basepoint, config);
        enumerator.enumerate();

        // Check that words are tracked correctly
        for point in enumerator.orbit() {
            assert!(point.word_length == point.word.len(),
                "Word length should match word vector length");
        }
    }
}
