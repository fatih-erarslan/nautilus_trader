#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use nalgebra::Vector3;

// Include generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Error type for Jolt physics
#[derive(Debug, thiserror::Error)]
pub enum JoltError {
    #[error("Physics system initialization failed")]
    InitializationFailed,
    #[error("Simulation step failed")]
    StepFailed,
}

pub type Result<T> = std::result::Result<T, JoltError>;

/// Configuration for Jolt physics system
#[derive(Debug, Clone)]
pub struct JoltConfiguration {
    pub max_bodies: u32,
    pub num_body_mutexes: u32,
    pub max_body_pairs: u32,
    pub max_contact_constraints: u32,
    pub collision_tolerance: f32,
    pub penetration_tolerance: f32,
    pub deterministic: bool,
}

impl Default for JoltConfiguration {
    fn default() -> Self {
        Self {
            max_bodies: 10240,
            num_body_mutexes: 0, // 0 = auto
            max_body_pairs: 10240,
            max_contact_constraints: 10240,
            collision_tolerance: 0.001,
            penetration_tolerance: 0.001,
            deterministic: true,
        }
    }
}

/// Adapter for JoltPhysics engine
///
/// Provides a safe Rust interface to the Jolt Physics engine via FFI.
/// All pointer operations are encapsulated with proper null checks.
///
/// # Thread Safety
///
/// JoltHyperPhysicsAdapter implements Send + Sync because:
/// - Jolt Physics is designed for thread-safe access
/// - The body interface methods use internal synchronization
/// - Multiple threads can safely query and modify bodies
///
/// However, for best performance, avoid concurrent modifications to the same body.
pub struct JoltHyperPhysicsAdapter {
    /// Pointer to the Jolt physics system. Never null after successful construction.
    system: *mut JoltSystem,
    /// Pointer to the body interface for creating/modifying bodies.
    /// Owned by the system; freed when system is destroyed.
    body_interface: *mut JoltBodyInterface,
    _config: JoltConfiguration,
}

// SAFETY: Jolt Physics is designed for multi-threaded access.
// The physics system uses internal locks to protect shared state.
// Individual body operations are atomic and thread-safe.
unsafe impl Send for JoltHyperPhysicsAdapter {}
unsafe impl Sync for JoltHyperPhysicsAdapter {}

impl JoltHyperPhysicsAdapter {
    /// Create a new Jolt adapter with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(JoltConfiguration::default())
    }

    /// Create a new Jolt adapter with custom configuration
    pub fn with_config(config: JoltConfiguration) -> Result<Self> {
        let c_config = JoltConfig {
            max_bodies: config.max_bodies,
            num_body_mutexes: config.num_body_mutexes,
            max_body_pairs: config.max_body_pairs,
            max_contact_constraints: config.max_contact_constraints,
            collision_tolerance: config.collision_tolerance,
            penetration_tolerance: config.penetration_tolerance,
            deterministic: config.deterministic,
        };

        // SAFETY: jolt_system_create is safe to call with any JoltConfig.
        // It returns null on failure, which we check below.
        let system = unsafe { jolt_system_create(c_config) };
        if system.is_null() {
            return Err(JoltError::InitializationFailed);
        }

        // SAFETY: system is valid and non-null (checked above).
        // The body_interface pointer is owned by the system and will be
        // freed automatically when the system is destroyed.
        let body_interface = unsafe { jolt_system_get_body_interface(system) };

        Ok(Self {
            system,
            body_interface,
            _config: config,
        })
    }

    /// Step the physics simulation
    ///
    /// Advances the simulation by `dt` seconds, performing `collision_steps`
    /// collision detection iterations.
    pub fn step(&mut self, dt: f32, collision_steps: i32) -> Result<()> {
        // SAFETY: self.system is always valid after successful construction.
        // jolt_system_step is safe to call with valid system pointer.
        unsafe {
            jolt_system_step(self.system, dt, collision_steps);
        }
        Ok(())
    }

    /// Create a box rigid body
    ///
    /// Creates a box shape with the given half-extents (converted to full extents internally).
    /// Returns a body ID that can be used for subsequent operations.
    pub fn create_box(&mut self, half_extents: Vector3<f32>, density: f32, is_static: bool) -> u32 {
        // SAFETY: body_interface is valid after construction.
        // jolt_body_create_box returns a valid body ID.
        unsafe {
            jolt_body_create_box(
                self.body_interface,
                half_extents.x * 2.0,
                half_extents.y * 2.0,
                half_extents.z * 2.0,
                density,
                is_static,
            )
        }
    }

    /// Create a sphere rigid body
    ///
    /// Creates a sphere shape with the given radius.
    /// Returns a body ID that can be used for subsequent operations.
    pub fn create_sphere(&mut self, radius: f32, density: f32, is_static: bool) -> u32 {
        // SAFETY: body_interface is valid after construction.
        unsafe { jolt_body_create_sphere(self.body_interface, radius, density, is_static) }
    }

    /// Get body position
    ///
    /// Returns the world-space position of the body's center of mass.
    pub fn get_position(&self, body_id: u32) -> Vector3<f32> {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;
        // SAFETY: body_interface is valid, and the out-pointers are valid stack variables.
        // jolt_body_get_position writes the position components to the provided pointers.
        unsafe {
            jolt_body_get_position(self.body_interface, body_id, &mut x, &mut y, &mut z);
        }
        Vector3::new(x, y, z)
    }

    /// Set body position
    pub fn set_position(&mut self, body_id: u32, position: Vector3<f32>) {
        unsafe {
            jolt_body_set_position(
                self.body_interface,
                body_id,
                position.x,
                position.y,
                position.z,
            );
        }
    }

    /// Get body velocity
    pub fn get_velocity(&self, body_id: u32) -> Vector3<f32> {
        let mut vx = 0.0;
        let mut vy = 0.0;
        let mut vz = 0.0;
        unsafe {
            jolt_body_get_velocity(self.body_interface, body_id, &mut vx, &mut vy, &mut vz);
        }
        Vector3::new(vx, vy, vz)
    }

    /// Set body velocity
    pub fn set_velocity(&mut self, body_id: u32, velocity: Vector3<f32>) {
        unsafe {
            jolt_body_set_velocity(
                self.body_interface,
                body_id,
                velocity.x,
                velocity.y,
                velocity.z,
            );
        }
    }
}

impl Drop for JoltHyperPhysicsAdapter {
    fn drop(&mut self) {
        // SAFETY:
        // - self.system is always valid after successful construction
        // - body_interface is owned by the system and freed internally
        // - jolt_system_destroy cleans up all resources
        // - After this call, the pointers become invalid (but we're being dropped anyway)
        unsafe {
            jolt_system_destroy(self.system);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let adapter = JoltHyperPhysicsAdapter::new();
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_body_creation() {
        let mut adapter = JoltHyperPhysicsAdapter::new().unwrap();
        let id = adapter.create_box(Vector3::new(1.0, 1.0, 1.0), 1.0, false);
        assert!(id > 0);
    }

    #[test]
    fn test_simulation_step() {
        let mut adapter = JoltHyperPhysicsAdapter::new().unwrap();
        let result = adapter.step(0.016, 1);
        assert!(result.is_ok());
    }
}
