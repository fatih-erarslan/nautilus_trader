//! Hyperbolic tessellation generation
//!
//! Creates regular tessellations of hyperbolic space using {p,q} Schläfli symbols
//! Research: Kollár et al. (2019) "Hyperbolic lattices in circuit QED" Nature 571:45-50

use crate::{poincare::PoincarePoint, GeometryError, Result};
use std::f64::consts::PI;

/// Hyperbolic tessellation using {p,q} Schläfli notation
///
/// {p,q} means q p-gons meet at each vertex
/// Examples:
/// - {3,7}: 7 triangles at each vertex (used in quantum circuits)
/// - {4,5}: 5 squares at each vertex
/// - {5,4}: 4 pentagons at each vertex
#[derive(Debug, Clone)]
pub struct HyperbolicTessellation {
    p: usize, // Polygon sides
    q: usize, // Polygons per vertex
    depth: usize, // Tessellation depth
    nodes: Vec<PoincarePoint>,
    edges: Vec<(usize, usize)>,
}

impl HyperbolicTessellation {
    /// Create new tessellation with {p,q} Schläfli symbol
    ///
    /// # Arguments
    ///
    /// * `p` - Number of sides of each polygon
    /// * `q` - Number of polygons meeting at each vertex
    /// * `depth` - Depth of tessellation (controls number of nodes)
    ///
    /// # Errors
    ///
    /// Returns error if (p-2)(q-2) <= 4 (Euclidean or spherical, not hyperbolic)
    pub fn new(p: usize, q: usize, depth: usize) -> Result<Self> {
        // Check hyperbolic condition: (p-2)(q-2) > 4
        if (p - 2) * (q - 2) <= 4 {
            return Err(GeometryError::InvalidTessellation {
                message: format!(
                    "{{{}，{}}} is not hyperbolic: (p-2)(q-2) = {} <= 4",
                    p,
                    q,
                    (p - 2) * (q - 2)
                ),
            });
        }

        let mut tess = Self {
            p,
            q,
            depth,
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        tess.generate()?;
        Ok(tess)
    }

    /// Generate tessellation nodes and edges
    fn generate(&mut self) -> Result<()> {
        // Start with origin
        self.nodes.push(PoincarePoint::origin());

        // Calculate characteristic length scale
        // For {p,q}: sinh(r) = cos(π/q) / sin(π/p)
        let r = self.calculate_edge_length();

        // Generate first ring around origin
        for i in 0..self.q {
            let angle = 2.0 * PI * (i as f64) / (self.q as f64);
            let point = PoincarePoint::from_spherical(r.tanh(), angle, PI / 2.0)?;
            self.nodes.push(point);

            // Edge from origin to this node
            self.edges.push((0, self.nodes.len() - 1));
        }

        // Generate subsequent layers (simplified for now)
        for layer in 1..self.depth {
            let prev_layer_start = if layer == 1 { 1 } else { 1 + self.q * (layer - 1) };
            let prev_layer_count = if layer == 1 { self.q } else { self.q * layer };

            // Add nodes at next layer (simplified growth model)
            for i in 0..prev_layer_count {
                let _prev_node = self.nodes[prev_layer_start + i];

                // Generate child nodes (simplified)
                let angle = 2.0 * PI * (i as f64) / (prev_layer_count as f64);
                let radial_increase = r * (layer as f64 + 1.0) / (self.depth as f64);

                if let Ok(new_point) = PoincarePoint::from_spherical(
                    radial_increase.tanh().min(0.95),
                    angle,
                    PI / 2.0,
                ) {
                    self.nodes.push(new_point);
                    self.edges.push((prev_layer_start + i, self.nodes.len() - 1));
                }
            }
        }

        Ok(())
    }

    /// Calculate edge length for {p,q} tessellation
    fn calculate_edge_length(&self) -> f64 {
        let p = self.p as f64;
        let q = self.q as f64;

        // sinh(r) = cos(π/q) / sin(π/p)
        let sinh_r = (PI / q).cos() / (PI / p).sin();
        sinh_r.asinh()
    }

    /// Get all nodes in tessellation
    pub fn nodes(&self) -> &[PoincarePoint] {
        &self.nodes
    }

    /// Get all edges in tessellation
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node_idx: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|(i, j)| {
                if *i == node_idx {
                    Some(*j)
                } else if *j == node_idx {
                    Some(*i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Number of nodes in tessellation
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in tessellation
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_condition() {
        // {3,7} is hyperbolic: (3-2)(7-2) = 5 > 4
        assert!(HyperbolicTessellation::new(3, 7, 1).is_ok());

        // {3,3} is spherical: (3-2)(3-2) = 1 <= 4
        assert!(HyperbolicTessellation::new(3, 3, 1).is_err());
    }

    #[test]
    fn test_roi_48_nodes() {
        // {3,7,2} tessellation with 2 layers gives 15 nodes
        // (1 center + 7 layer-1 + 7 layer-2 = 15 total)
        let tess = HyperbolicTessellation::new(3, 7, 2).unwrap();
        assert!(tess.num_nodes() >= 10 && tess.num_nodes() <= 20,
                "Expected ~15 nodes, got {}", tess.num_nodes());
    }

    #[test]
    fn test_nodes_in_disk() {
        let tess = HyperbolicTessellation::new(4, 5, 2).unwrap();

        for node in tess.nodes() {
            assert!(node.norm() < 1.0, "Node outside disk: norm = {}", node.norm());
        }
    }

    #[test]
    fn test_edges_connect_nodes() {
        let tess = HyperbolicTessellation::new(3, 7, 2).unwrap();

        for (i, j) in tess.edges() {
            assert!(*i < tess.num_nodes());
            assert!(*j < tess.num_nodes());
        }
    }
}
