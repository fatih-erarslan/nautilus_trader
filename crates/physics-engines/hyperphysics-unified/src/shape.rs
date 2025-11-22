//! Collision shape definitions

use crate::AABB;
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

/// Collision shape types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Shape {
    /// Sphere shape
    Sphere { radius: f32 },

    /// Box (cuboid) shape
    Box { half_extents: Vector3<f32> },

    /// Capsule shape (cylinder with hemispherical caps)
    Capsule {
        half_height: f32,
        radius: f32,
    },

    /// Cylinder shape
    Cylinder {
        half_height: f32,
        radius: f32,
    },

    /// Cone shape
    Cone {
        half_height: f32,
        radius: f32,
    },

    /// Convex hull from points
    ConvexHull { points: Vec<Point3<f32>> },

    /// Triangle mesh (static only)
    TriMesh {
        vertices: Vec<Point3<f32>>,
        indices: Vec<[u32; 3]>,
    },

    /// Height field terrain
    HeightField {
        heights: Vec<f32>,
        rows: u32,
        cols: u32,
        scale: Vector3<f32>,
    },

    /// Compound shape (multiple sub-shapes)
    Compound {
        shapes: Vec<(Shape, crate::Transform)>,
    },
}

impl Shape {
    /// Create a sphere shape
    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create a box shape from half-extents
    pub fn cuboid(half_extents: Vector3<f32>) -> Self {
        Self::Box { half_extents }
    }

    /// Create a box shape from full dimensions
    pub fn cuboid_from_size(size: Vector3<f32>) -> Self {
        Self::Box {
            half_extents: size * 0.5,
        }
    }

    /// Create a capsule shape
    pub fn capsule(half_height: f32, radius: f32) -> Self {
        Self::Capsule { half_height, radius }
    }

    /// Create a cylinder shape
    pub fn cylinder(half_height: f32, radius: f32) -> Self {
        Self::Cylinder { half_height, radius }
    }

    /// Create a cone shape
    pub fn cone(half_height: f32, radius: f32) -> Self {
        Self::Cone { half_height, radius }
    }

    /// Create a convex hull from points
    pub fn convex_hull(points: Vec<Point3<f32>>) -> Self {
        Self::ConvexHull { points }
    }

    /// Create a triangle mesh
    pub fn trimesh(vertices: Vec<Point3<f32>>, indices: Vec<[u32; 3]>) -> Self {
        Self::TriMesh { vertices, indices }
    }

    /// Create a height field
    pub fn heightfield(heights: Vec<f32>, rows: u32, cols: u32, scale: Vector3<f32>) -> Self {
        Self::HeightField {
            heights,
            rows,
            cols,
            scale,
        }
    }

    /// Create a compound shape
    pub fn compound(shapes: Vec<(Shape, crate::Transform)>) -> Self {
        Self::Compound { shapes }
    }

    /// Compute local AABB of this shape
    pub fn local_aabb(&self) -> AABB {
        match self {
            Shape::Sphere { radius } => {
                let r = Vector3::new(*radius, *radius, *radius);
                AABB::new(Point3::origin() - r, Point3::origin() + r)
            }
            Shape::Box { half_extents } => {
                AABB::new(Point3::origin() - half_extents, Point3::origin() + half_extents)
            }
            Shape::Capsule { half_height, radius } => {
                let he = Vector3::new(*radius, *half_height + *radius, *radius);
                AABB::new(Point3::origin() - he, Point3::origin() + he)
            }
            Shape::Cylinder { half_height, radius } => {
                let he = Vector3::new(*radius, *half_height, *radius);
                AABB::new(Point3::origin() - he, Point3::origin() + he)
            }
            Shape::Cone { half_height, radius } => {
                let he = Vector3::new(*radius, *half_height, *radius);
                AABB::new(Point3::origin() - he, Point3::origin() + he)
            }
            Shape::ConvexHull { points } => {
                let mut min = Point3::new(f32::MAX, f32::MAX, f32::MAX);
                let mut max = Point3::new(f32::MIN, f32::MIN, f32::MIN);
                for p in points {
                    min.x = min.x.min(p.x);
                    min.y = min.y.min(p.y);
                    min.z = min.z.min(p.z);
                    max.x = max.x.max(p.x);
                    max.y = max.y.max(p.y);
                    max.z = max.z.max(p.z);
                }
                AABB::new(min, max)
            }
            Shape::TriMesh { vertices, .. } => {
                let mut min = Point3::new(f32::MAX, f32::MAX, f32::MAX);
                let mut max = Point3::new(f32::MIN, f32::MIN, f32::MIN);
                for p in vertices {
                    min.x = min.x.min(p.x);
                    min.y = min.y.min(p.y);
                    min.z = min.z.min(p.z);
                    max.x = max.x.max(p.x);
                    max.y = max.y.max(p.y);
                    max.z = max.z.max(p.z);
                }
                AABB::new(min, max)
            }
            Shape::HeightField {
                heights,
                rows,
                cols,
                scale,
            } => {
                let min_h = heights.iter().cloned().fold(f32::MAX, f32::min);
                let max_h = heights.iter().cloned().fold(f32::MIN, f32::max);
                AABB::new(
                    Point3::new(0.0, min_h * scale.y, 0.0),
                    Point3::new(
                        (*cols as f32 - 1.0) * scale.x,
                        max_h * scale.y,
                        (*rows as f32 - 1.0) * scale.z,
                    ),
                )
            }
            Shape::Compound { shapes } => {
                let mut result = AABB::default();
                for (shape, transform) in shapes {
                    let local_aabb = shape.local_aabb();
                    // Transform AABB corners (conservative)
                    let corners = [
                        Point3::new(local_aabb.min.x, local_aabb.min.y, local_aabb.min.z),
                        Point3::new(local_aabb.max.x, local_aabb.min.y, local_aabb.min.z),
                        Point3::new(local_aabb.min.x, local_aabb.max.y, local_aabb.min.z),
                        Point3::new(local_aabb.max.x, local_aabb.max.y, local_aabb.min.z),
                        Point3::new(local_aabb.min.x, local_aabb.min.y, local_aabb.max.z),
                        Point3::new(local_aabb.max.x, local_aabb.min.y, local_aabb.max.z),
                        Point3::new(local_aabb.min.x, local_aabb.max.y, local_aabb.max.z),
                        Point3::new(local_aabb.max.x, local_aabb.max.y, local_aabb.max.z),
                    ];
                    for corner in &corners {
                        let transformed = transform.transform_point(corner);
                        result.min.x = result.min.x.min(transformed.x);
                        result.min.y = result.min.y.min(transformed.y);
                        result.min.z = result.min.z.min(transformed.z);
                        result.max.x = result.max.x.max(transformed.x);
                        result.max.y = result.max.y.max(transformed.y);
                        result.max.z = result.max.z.max(transformed.z);
                    }
                }
                result
            }
        }
    }

    /// Compute volume of shape
    pub fn volume(&self) -> f32 {
        match self {
            Shape::Sphere { radius } => (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3),
            Shape::Box { half_extents } => 8.0 * half_extents.x * half_extents.y * half_extents.z,
            Shape::Capsule { half_height, radius } => {
                let cylinder = std::f32::consts::PI * radius.powi(2) * (2.0 * half_height);
                let sphere = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
                cylinder + sphere
            }
            Shape::Cylinder { half_height, radius } => {
                std::f32::consts::PI * radius.powi(2) * (2.0 * half_height)
            }
            Shape::Cone { half_height, radius } => {
                (1.0 / 3.0) * std::f32::consts::PI * radius.powi(2) * (2.0 * half_height)
            }
            Shape::ConvexHull { .. } => {
                // Approximate with AABB volume
                self.local_aabb().volume()
            }
            Shape::TriMesh { .. } => {
                // Would need signed volume calculation
                0.0
            }
            Shape::HeightField { .. } => 0.0, // Not meaningful
            Shape::Compound { shapes } => shapes.iter().map(|(s, _)| s.volume()).sum(),
        }
    }

    /// Check if shape is convex
    pub fn is_convex(&self) -> bool {
        match self {
            Shape::Sphere { .. }
            | Shape::Box { .. }
            | Shape::Capsule { .. }
            | Shape::Cylinder { .. }
            | Shape::Cone { .. }
            | Shape::ConvexHull { .. } => true,
            Shape::TriMesh { .. } | Shape::HeightField { .. } => false,
            Shape::Compound { shapes } => shapes.iter().all(|(s, _)| s.is_convex()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_aabb() {
        let shape = Shape::sphere(2.0);
        let aabb = shape.local_aabb();
        assert_eq!(aabb.min, Point3::new(-2.0, -2.0, -2.0));
        assert_eq!(aabb.max, Point3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_sphere_volume() {
        let shape = Shape::sphere(1.0);
        let expected = (4.0 / 3.0) * std::f32::consts::PI;
        assert!((shape.volume() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_box_volume() {
        let shape = Shape::cuboid(Vector3::new(1.0, 2.0, 3.0));
        assert!((shape.volume() - 48.0).abs() < 1e-6); // 2*4*6 = 48
    }
}
