//! # HyperPhysics Unified Physics Engine
//!
//! Zero-cost abstraction layer for multiple physics engine backends with:
//! - **Memory-efficient object pooling** - Arena allocation, generational indices
//! - **SIMD-accelerated broadphase** - AVX2/NEON collision culling
//! - **Deterministic simulation** - Bit-exact reproducibility
//! - **Backend agnostic** - Rapier, Box2D, Parry, custom engines
//!
//! ## Performance Targets
//! - Broadphase: <10μs for 10K bodies
//! - Narrowphase: <50μs for 1K contact pairs
//! - Memory: <64 bytes per static body, <128 bytes per dynamic body
//!
//! ## Usage
//! ```ignore
//! use hyperphysics_unified::prelude::*;
//!
//! let mut world = PhysicsWorld::<RapierBackend>::new(WorldConfig::default());
//! let body = world.create_body(BodyDesc::dynamic().with_shape(Shape::sphere(1.0)));
//! world.step(1.0 / 60.0);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod backend;
pub mod body;
pub mod broadphase;
pub mod collider;
pub mod constraint;
pub mod memory;
pub mod query;
pub mod shape;
pub mod world;

pub mod prelude {
    //! Convenience re-exports
    pub use crate::backend::PhysicsBackend;
    pub use crate::body::{BodyDesc, BodyHandle, BodyType, RigidBody};
    pub use crate::broadphase::BroadPhase;
    pub use crate::collider::{Collider, ColliderHandle};
    pub use crate::constraint::{Constraint, ConstraintHandle};
    pub use crate::memory::{Arena, ObjectPool};
    pub use crate::query::{RayCast, RayHit, ShapeCast};
    pub use crate::shape::Shape;
    pub use crate::world::{PhysicsWorld, WorldConfig};

    #[cfg(feature = "rapier3d")]
    pub use crate::backend::rapier::RapierBackend;
}

use nalgebra::{Isometry3, Point3, Quaternion, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// 3D transform (position + rotation)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    /// Position in world space
    pub position: Vector3<f32>,
    /// Rotation as unit quaternion
    pub rotation: UnitQuaternion<f32>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            rotation: UnitQuaternion::identity(),
        }
    }
}

impl Transform {
    /// Create transform from position only
    pub fn from_position(position: Vector3<f32>) -> Self {
        Self {
            position,
            rotation: UnitQuaternion::identity(),
        }
    }

    /// Create transform from position and axis-angle rotation
    pub fn from_position_axis_angle(position: Vector3<f32>, axis: Vector3<f32>, angle: f32) -> Self {
        Self {
            position,
            rotation: UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(axis),
                angle,
            ),
        }
    }

    /// Convert to nalgebra Isometry3
    pub fn to_isometry(&self) -> Isometry3<f32> {
        Isometry3::from_parts(self.position.into(), self.rotation)
    }

    /// Create from nalgebra Isometry3
    pub fn from_isometry(iso: &Isometry3<f32>) -> Self {
        Self {
            position: iso.translation.vector,
            rotation: iso.rotation,
        }
    }

    /// Transform a point from local to world space
    pub fn transform_point(&self, point: &Point3<f32>) -> Point3<f32> {
        self.rotation * point + self.position
    }

    /// Transform a vector (direction) from local to world space
    pub fn transform_vector(&self, vector: &Vector3<f32>) -> Vector3<f32> {
        self.rotation * vector
    }

    /// Inverse transform a point from world to local space
    pub fn inverse_transform_point(&self, point: &Point3<f32>) -> Point3<f32> {
        self.rotation.inverse() * (point - self.position)
    }

    /// Interpolate between two transforms
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            position: self.position.lerp(&other.position, t),
            rotation: self.rotation.slerp(&other.rotation, t),
        }
    }
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AABB {
    /// Minimum corner
    pub min: Point3<f32>,
    /// Maximum corner
    pub max: Point3<f32>,
}

impl AABB {
    /// Create AABB from min/max corners
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }

    /// Create AABB from center and half-extents
    pub fn from_center_half_extents(center: Point3<f32>, half_extents: Vector3<f32>) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Get AABB center
    pub fn center(&self) -> Point3<f32> {
        nalgebra::center(&self.min, &self.max)
    }

    /// Get AABB half-extents
    pub fn half_extents(&self) -> Vector3<f32> {
        (self.max - self.min) * 0.5
    }

    /// Get AABB extents (full size)
    pub fn extents(&self) -> Vector3<f32> {
        self.max - self.min
    }

    /// Check if two AABBs intersect
    #[inline]
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Check if AABB contains a point
    #[inline]
    pub fn contains(&self, point: &Point3<f32>) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Compute union of two AABBs
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Expand AABB by a margin
    pub fn expand(&self, margin: f32) -> AABB {
        let margin_vec = Vector3::new(margin, margin, margin);
        AABB {
            min: self.min - margin_vec,
            max: self.max + margin_vec,
        }
    }

    /// Surface area (for SAH in BVH)
    pub fn surface_area(&self) -> f32 {
        let d = self.extents();
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    /// Volume
    pub fn volume(&self) -> f32 {
        let d = self.extents();
        d.x * d.y * d.z
    }
}

impl Default for AABB {
    fn default() -> Self {
        Self {
            min: Point3::new(f32::MAX, f32::MAX, f32::MAX),
            max: Point3::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }
}

/// Physics material properties
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhysicsMaterial {
    /// Coefficient of friction (0.0 = frictionless, 1.0 = very rough)
    pub friction: f32,
    /// Coefficient of restitution (0.0 = inelastic, 1.0 = perfectly elastic)
    pub restitution: f32,
    /// Density in kg/m³
    pub density: f32,
}

impl Default for PhysicsMaterial {
    fn default() -> Self {
        Self {
            friction: 0.5,
            restitution: 0.3,
            density: 1000.0, // Water density
        }
    }
}

impl PhysicsMaterial {
    /// Steel material
    pub const STEEL: Self = Self {
        friction: 0.4,
        restitution: 0.6,
        density: 7800.0,
    };

    /// Rubber material
    pub const RUBBER: Self = Self {
        friction: 0.9,
        restitution: 0.8,
        density: 1100.0,
    };

    /// Ice material
    pub const ICE: Self = Self {
        friction: 0.02,
        restitution: 0.1,
        density: 917.0,
    };

    /// Wood material
    pub const WOOD: Self = Self {
        friction: 0.4,
        restitution: 0.4,
        density: 700.0,
    };
}

/// Contact point information
#[derive(Debug, Clone, Copy)]
pub struct ContactPoint {
    /// World-space contact position
    pub position: Point3<f32>,
    /// Contact normal (from body A to body B)
    pub normal: Vector3<f32>,
    /// Penetration depth (negative = separated)
    pub depth: f32,
    /// Impulse applied at this contact
    pub impulse: f32,
}

/// Contact manifold between two bodies
#[derive(Debug, Clone)]
pub struct ContactManifold {
    /// First body handle
    pub body_a: body::BodyHandle,
    /// Second body handle
    pub body_b: body::BodyHandle,
    /// Contact points (max 4 for stability)
    pub points: smallvec::SmallVec<[ContactPoint; 4]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform() {
        let t = Transform::from_position(Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(t.position, Vector3::new(1.0, 2.0, 3.0));

        let point = Point3::new(1.0, 0.0, 0.0);
        let transformed = t.transform_point(&point);
        assert!((transformed - Point3::new(2.0, 2.0, 3.0)).norm() < 1e-6);
    }

    #[test]
    fn test_aabb_intersection() {
        let a = AABB::from_center_half_extents(Point3::origin(), Vector3::new(1.0, 1.0, 1.0));
        let b = AABB::from_center_half_extents(
            Point3::new(1.5, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );
        let c = AABB::from_center_half_extents(
            Point3::new(5.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }
}
