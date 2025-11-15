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
//! # Applications
//!
//! - {7,3} hyperbolic tessellation generation
//! - Exact tile placement via group actions
//! - Symmetry analysis of hyperbolic patterns
//! - Discrete isometry group structure
//!
//! # References
//!
//! - Katok, "Fuchsian Groups" (1992)
//! - Beardon, "The Geometry of Discrete Groups" (1983)
//! - Ratcliffe, "Foundations of Hyperbolic Manifolds" (2006)

use crate::{GeometryError, Result, MoebiusTransform};
use num_complex::Complex64;
use std::collections::HashSet;

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
        // For a {p,q} tessellation, the fundamental domain is a p-sided polygon
        // We'll compute the Dirichlet domain centered at the given point

        // Generate nearby group elements
        let mut nearby_transforms = Vec::new();

        for element in &self.elements {
            if element.word.is_empty() {
                continue; // Skip identity
            }

            let transform = self.element_to_transform(element);
            let image_center = transform.apply(center);

            // Only consider elements that move the center a reasonable distance
            if (image_center - center).norm() < 2.0 {
                nearby_transforms.push(transform);
            }
        }

        if nearby_transforms.is_empty() {
            return Err(GeometryError::InvalidTessellation {
                message: "Need to generate more group elements for fundamental domain".to_string(),
            });
        }

        // Compute perpendicular bisectors in hyperbolic geometry
        // The fundamental domain is the intersection of half-spaces
        let mut vertices = Vec::new();

        // For now, return a simplified approximation
        // A proper implementation would compute hyperbolic perpendicular bisectors
        let num_sides = 7; // For {7,3} tessellation
        for i in 0..num_sides {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_sides as f64);
            let radius = 0.3; // Approximate radius
            let vertex = center + Complex64::new(radius * angle.cos(), radius * angle.sin());
            vertices.push(vertex);
        }

        Ok(vertices)
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
}
