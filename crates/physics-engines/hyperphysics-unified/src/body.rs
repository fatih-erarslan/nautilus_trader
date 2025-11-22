//! Rigid body definitions and management

use crate::{PhysicsMaterial, Transform, AABB};
use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};
use slotmap::new_key_type;

new_key_type! {
    /// Handle to a rigid body in the physics world
    pub struct BodyHandle;
}

/// Type of rigid body
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BodyType {
    /// Static body - never moves, infinite mass
    Static,
    /// Dynamic body - fully simulated
    Dynamic,
    /// Kinematic body - moved by user, affects dynamic bodies
    Kinematic,
}

impl Default for BodyType {
    fn default() -> Self {
        Self::Dynamic
    }
}

/// Rigid body description for creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyDesc {
    /// Body type
    pub body_type: BodyType,
    /// Initial transform
    pub transform: Transform,
    /// Initial linear velocity
    pub linear_velocity: Vector3<f32>,
    /// Initial angular velocity
    pub angular_velocity: Vector3<f32>,
    /// Linear damping (0 = no damping)
    pub linear_damping: f32,
    /// Angular damping (0 = no damping)
    pub angular_damping: f32,
    /// Gravity scale (0 = no gravity, 1 = normal, negative = anti-gravity)
    pub gravity_scale: f32,
    /// Can this body sleep when inactive?
    pub can_sleep: bool,
    /// Is CCD enabled for this body?
    pub ccd_enabled: bool,
    /// User data (opaque identifier)
    pub user_data: u64,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            transform: Transform::default(),
            linear_velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            linear_damping: 0.0,
            angular_damping: 0.05,
            gravity_scale: 1.0,
            can_sleep: true,
            ccd_enabled: false,
            user_data: 0,
        }
    }
}

impl BodyDesc {
    /// Create a static body description
    pub fn fixed() -> Self {
        Self {
            body_type: BodyType::Static,
            ..Default::default()
        }
    }

    /// Create a dynamic body description
    pub fn dynamic() -> Self {
        Self {
            body_type: BodyType::Dynamic,
            ..Default::default()
        }
    }

    /// Create a kinematic body description
    pub fn kinematic() -> Self {
        Self {
            body_type: BodyType::Kinematic,
            ..Default::default()
        }
    }

    /// Set initial position
    pub fn with_position(mut self, position: Vector3<f32>) -> Self {
        self.transform.position = position;
        self
    }

    /// Set initial transform
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Set initial linear velocity
    pub fn with_linear_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.linear_velocity = velocity;
        self
    }

    /// Set initial angular velocity
    pub fn with_angular_velocity(mut self, velocity: Vector3<f32>) -> Self {
        self.angular_velocity = velocity;
        self
    }

    /// Set linear damping
    pub fn with_linear_damping(mut self, damping: f32) -> Self {
        self.linear_damping = damping;
        self
    }

    /// Set angular damping
    pub fn with_angular_damping(mut self, damping: f32) -> Self {
        self.angular_damping = damping;
        self
    }

    /// Set gravity scale
    pub fn with_gravity_scale(mut self, scale: f32) -> Self {
        self.gravity_scale = scale;
        self
    }

    /// Enable/disable sleeping
    pub fn with_can_sleep(mut self, can_sleep: bool) -> Self {
        self.can_sleep = can_sleep;
        self
    }

    /// Enable CCD for fast-moving objects
    pub fn with_ccd(mut self, enabled: bool) -> Self {
        self.ccd_enabled = enabled;
        self
    }

    /// Set user data
    pub fn with_user_data(mut self, data: u64) -> Self {
        self.user_data = data;
        self
    }
}

/// Mass properties of a rigid body
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MassProperties {
    /// Total mass in kg
    pub mass: f32,
    /// Inverse mass (0 for infinite mass)
    pub inv_mass: f32,
    /// Center of mass in local space
    pub center_of_mass: Vector3<f32>,
    /// Inertia tensor in local space
    pub inertia: Matrix3<f32>,
    /// Inverse inertia tensor
    pub inv_inertia: Matrix3<f32>,
}

impl MassProperties {
    /// Create mass properties for a sphere
    pub fn sphere(mass: f32, radius: f32) -> Self {
        let i = 0.4 * mass * radius * radius;
        let inertia = Matrix3::from_diagonal(&Vector3::new(i, i, i));
        Self::from_mass_inertia(mass, Vector3::zeros(), inertia)
    }

    /// Create mass properties for a box
    pub fn cuboid(mass: f32, half_extents: Vector3<f32>) -> Self {
        let x2 = half_extents.x * half_extents.x * 4.0;
        let y2 = half_extents.y * half_extents.y * 4.0;
        let z2 = half_extents.z * half_extents.z * 4.0;
        let factor = mass / 12.0;
        let inertia = Matrix3::from_diagonal(&Vector3::new(
            factor * (y2 + z2),
            factor * (x2 + z2),
            factor * (x2 + y2),
        ));
        Self::from_mass_inertia(mass, Vector3::zeros(), inertia)
    }

    /// Create mass properties for a cylinder
    pub fn cylinder(mass: f32, half_height: f32, radius: f32) -> Self {
        let h2 = half_height * half_height * 4.0;
        let r2 = radius * radius;
        let iy = 0.5 * mass * r2;
        let ixz = mass * (3.0 * r2 + h2) / 12.0;
        let inertia = Matrix3::from_diagonal(&Vector3::new(ixz, iy, ixz));
        Self::from_mass_inertia(mass, Vector3::zeros(), inertia)
    }

    /// Create from mass, center of mass, and inertia tensor
    pub fn from_mass_inertia(mass: f32, com: Vector3<f32>, inertia: Matrix3<f32>) -> Self {
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        let inv_inertia = if mass > 0.0 {
            inertia.try_inverse().unwrap_or(Matrix3::zeros())
        } else {
            Matrix3::zeros()
        };

        Self {
            mass,
            inv_mass,
            center_of_mass: com,
            inertia,
            inv_inertia,
        }
    }

    /// Infinite mass (static body)
    pub fn infinite() -> Self {
        Self {
            mass: f32::INFINITY,
            inv_mass: 0.0,
            center_of_mass: Vector3::zeros(),
            inertia: Matrix3::zeros(),
            inv_inertia: Matrix3::zeros(),
        }
    }
}

/// Runtime state of a rigid body
#[derive(Debug, Clone)]
pub struct RigidBody {
    /// Body handle
    pub handle: BodyHandle,
    /// Body type
    pub body_type: BodyType,
    /// Current transform
    pub transform: Transform,
    /// Linear velocity
    pub linear_velocity: Vector3<f32>,
    /// Angular velocity
    pub angular_velocity: Vector3<f32>,
    /// Mass properties
    pub mass_properties: MassProperties,
    /// Linear damping
    pub linear_damping: f32,
    /// Angular damping
    pub angular_damping: f32,
    /// Gravity scale
    pub gravity_scale: f32,
    /// Accumulated force (cleared after integration)
    pub force: Vector3<f32>,
    /// Accumulated torque (cleared after integration)
    pub torque: Vector3<f32>,
    /// Is body sleeping?
    pub is_sleeping: bool,
    /// Sleep counter (frames of low motion)
    pub sleep_counter: u32,
    /// AABB (updated after integration)
    pub aabb: AABB,
    /// User data
    pub user_data: u64,
}

impl RigidBody {
    /// Create from description
    pub fn from_desc(handle: BodyHandle, desc: &BodyDesc, mass_props: MassProperties) -> Self {
        Self {
            handle,
            body_type: desc.body_type,
            transform: desc.transform,
            linear_velocity: desc.linear_velocity,
            angular_velocity: desc.angular_velocity,
            mass_properties: mass_props,
            linear_damping: desc.linear_damping,
            angular_damping: desc.angular_damping,
            gravity_scale: desc.gravity_scale,
            force: Vector3::zeros(),
            torque: Vector3::zeros(),
            is_sleeping: false,
            sleep_counter: 0,
            aabb: AABB::default(),
            user_data: desc.user_data,
        }
    }

    /// Check if body is dynamic
    pub fn is_dynamic(&self) -> bool {
        self.body_type == BodyType::Dynamic
    }

    /// Check if body is static
    pub fn is_static(&self) -> bool {
        self.body_type == BodyType::Static
    }

    /// Check if body is kinematic
    pub fn is_kinematic(&self) -> bool {
        self.body_type == BodyType::Kinematic
    }

    /// Get world-space center of mass
    pub fn world_com(&self) -> Vector3<f32> {
        self.transform.position + self.transform.rotation * self.mass_properties.center_of_mass
    }

    /// Apply force at center of mass
    pub fn apply_force(&mut self, force: Vector3<f32>) {
        if self.is_dynamic() {
            self.force += force;
            self.wake_up();
        }
    }

    /// Apply force at world point
    pub fn apply_force_at_point(&mut self, force: Vector3<f32>, point: Vector3<f32>) {
        if self.is_dynamic() {
            self.force += force;
            let r = point - self.world_com();
            self.torque += r.cross(&force);
            self.wake_up();
        }
    }

    /// Apply impulse at center of mass
    pub fn apply_impulse(&mut self, impulse: Vector3<f32>) {
        if self.is_dynamic() {
            self.linear_velocity += impulse * self.mass_properties.inv_mass;
            self.wake_up();
        }
    }

    /// Apply angular impulse
    pub fn apply_angular_impulse(&mut self, impulse: Vector3<f32>) {
        if self.is_dynamic() {
            self.angular_velocity += self.mass_properties.inv_inertia * impulse;
            self.wake_up();
        }
    }

    /// Wake up sleeping body
    pub fn wake_up(&mut self) {
        self.is_sleeping = false;
        self.sleep_counter = 0;
    }

    /// Put body to sleep
    pub fn sleep(&mut self) {
        self.is_sleeping = true;
        self.linear_velocity = Vector3::zeros();
        self.angular_velocity = Vector3::zeros();
    }

    /// Kinetic energy
    pub fn kinetic_energy(&self) -> f32 {
        let linear = 0.5 * self.mass_properties.mass * self.linear_velocity.norm_squared();
        let angular = 0.5 * self.angular_velocity.dot(&(self.mass_properties.inertia * self.angular_velocity));
        linear + angular
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_properties_sphere() {
        let mp = MassProperties::sphere(1.0, 1.0);
        assert!((mp.mass - 1.0).abs() < 1e-6);
        assert!((mp.inv_mass - 1.0).abs() < 1e-6);
        // I = 0.4 * m * r^2 = 0.4 for unit sphere
        assert!((mp.inertia[(0, 0)] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_body_desc_builder() {
        let desc = BodyDesc::dynamic()
            .with_position(Vector3::new(1.0, 2.0, 3.0))
            .with_linear_velocity(Vector3::new(0.0, 1.0, 0.0))
            .with_gravity_scale(0.5);

        assert_eq!(desc.body_type, BodyType::Dynamic);
        assert_eq!(desc.transform.position, Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(desc.linear_velocity, Vector3::new(0.0, 1.0, 0.0));
        assert!((desc.gravity_scale - 0.5).abs() < 1e-6);
    }
}
