//! Collider definitions

use crate::{shape::Shape, PhysicsMaterial, Transform, AABB};
use serde::{Deserialize, Serialize};
use slotmap::new_key_type;

new_key_type! {
    /// Handle to a collider
    pub struct ColliderHandle;
}

/// Collider description for creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColliderDesc {
    /// Collision shape
    pub shape: Shape,
    /// Local transform relative to parent body
    pub local_transform: Transform,
    /// Physics material
    pub material: PhysicsMaterial,
    /// Is sensor (no contact response)
    pub is_sensor: bool,
    /// Collision groups (bitmask)
    pub collision_groups: u32,
    /// Collision filter (bitmask)
    pub collision_filter: u32,
    /// User data
    pub user_data: u64,
}

impl ColliderDesc {
    /// Create collider with shape
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            local_transform: Transform::default(),
            material: PhysicsMaterial::default(),
            is_sensor: false,
            collision_groups: 0xFFFFFFFF,
            collision_filter: 0xFFFFFFFF,
            user_data: 0,
        }
    }

    /// Set local transform
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.local_transform = transform;
        self
    }

    /// Set material
    pub fn with_material(mut self, material: PhysicsMaterial) -> Self {
        self.material = material;
        self
    }

    /// Make this a sensor
    pub fn as_sensor(mut self) -> Self {
        self.is_sensor = true;
        self
    }

    /// Set collision groups
    pub fn with_groups(mut self, groups: u32, filter: u32) -> Self {
        self.collision_groups = groups;
        self.collision_filter = filter;
        self
    }
}

/// Runtime collider state
#[derive(Debug, Clone)]
pub struct Collider {
    /// Collider handle
    pub handle: ColliderHandle,
    /// Parent body handle
    pub parent: crate::body::BodyHandle,
    /// Collision shape
    pub shape: Shape,
    /// Local transform
    pub local_transform: Transform,
    /// World transform (computed)
    pub world_transform: Transform,
    /// Material
    pub material: PhysicsMaterial,
    /// Is sensor
    pub is_sensor: bool,
    /// Collision groups
    pub collision_groups: u32,
    /// Collision filter
    pub collision_filter: u32,
    /// World AABB
    pub aabb: AABB,
    /// Is enabled
    pub enabled: bool,
    /// User data
    pub user_data: u64,
}

impl Collider {
    /// Check if two colliders can collide based on groups
    pub fn can_collide_with(&self, other: &Collider) -> bool {
        (self.collision_groups & other.collision_filter) != 0
            && (other.collision_groups & self.collision_filter) != 0
    }
}
