//! Spatial queries (raycasting, shape casting)

use crate::shape::Shape;
use nalgebra::{Point3, UnitVector3, Vector3};

/// Ray cast query
#[derive(Debug, Clone)]
pub struct RayCast {
    /// Ray origin
    pub origin: Point3<f32>,
    /// Ray direction (normalized)
    pub direction: UnitVector3<f32>,
    /// Maximum distance
    pub max_distance: f32,
    /// Collision filter
    pub filter: u32,
}

impl RayCast {
    /// Create a new ray cast
    pub fn new(origin: Point3<f32>, direction: Vector3<f32>, max_distance: f32) -> Self {
        Self {
            origin,
            direction: UnitVector3::new_normalize(direction),
            max_distance,
            filter: 0xFFFFFFFF,
        }
    }
}

/// Ray cast hit result
#[derive(Debug, Clone)]
pub struct RayHit<H> {
    /// Body that was hit
    pub body: H,
    /// Hit point in world space
    pub point: Point3<f32>,
    /// Hit normal
    pub normal: Vector3<f32>,
    /// Distance from ray origin
    pub distance: f32,
}

/// Shape cast query
#[derive(Debug, Clone)]
pub struct ShapeCast {
    /// Shape to cast
    pub shape: Shape,
    /// Start position
    pub start: Point3<f32>,
    /// Cast direction
    pub direction: UnitVector3<f32>,
    /// Maximum distance
    pub max_distance: f32,
    /// Filter
    pub filter: u32,
}

/// Shape cast hit result
#[derive(Debug, Clone)]
pub struct ShapeHit<H> {
    /// Body that was hit
    pub body: H,
    /// Contact point
    pub point: Point3<f32>,
    /// Contact normal
    pub normal: Vector3<f32>,
    /// Time of impact (0-1)
    pub toi: f32,
}
