//! Avian (Bevy) physics backend
//!
//! ECS-native physics for Bevy game engine.
//! Fork: https://github.com/fatih-erarslan/avian
//!
//! ## Important Note
//!
//! Avian is tightly integrated with Bevy ECS and cannot be used standalone.
//! This backend provides a bridge layer that allows interoperability between
//! the HyperPhysics unified API and Bevy/Avian applications.
//!
//! ## Usage Modes
//!
//! ### Mode 1: Standalone (Limited)
//! Creates an internal Bevy World for physics simulation. Useful for
//! headless physics testing but doesn't integrate with a full Bevy app.
//!
//! ### Mode 2: Bevy Integration (Recommended)
//! Use `AvianBridge` to connect HyperPhysics with an existing Bevy World.
//! This allows full access to Avian's ECS features.
//!
//! ## Example (Standalone)
//! ```rust,ignore
//! let config = AvianConfig {
//!     gravity: Vector3::new(0.0, -9.81, 0.0),
//!     substeps: 4,
//!     ..Default::default()
//! };
//! let mut backend = AvianBackend::new(config)?;
//!
//! // Create bodies
//! let body = backend.create_body(&BodyDesc::dynamic())?;
//! backend.step(1.0 / 60.0);
//! ```
//!
//! ## Example (Bevy Integration)
//! ```rust,ignore
//! // In your Bevy app
//! fn setup(mut commands: Commands) {
//!     // Spawn rigid body with Avian components
//!     commands.spawn((
//!         RigidBody::Dynamic,
//!         Collider::sphere(0.5),
//!         Transform::from_xyz(0.0, 5.0, 0.0),
//!     ));
//! }
//!
//! // Use HyperPhysics bridge for unified API
//! fn physics_system(world: &mut World) {
//!     let bridge = AvianBridge::from_world(world);
//!     let body_count = bridge.body_count();
//! }
//! ```

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::{BodyDesc, BodyType};
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use std::any::Any;
use std::collections::HashMap;

#[cfg(feature = "avian3d")]
use avian3d::prelude::*;
#[cfg(feature = "avian3d")]
use bevy_ecs::prelude::*;
#[cfg(feature = "avian3d")]
use bevy_ecs::world::World;
#[cfg(feature = "avian3d")]
use bevy_app::{App, ScheduleRunnerPlugin};
#[cfg(feature = "avian3d")]
use bevy_time::{Time, TimePlugin};

/// Avian configuration
#[derive(Debug, Clone)]
pub struct AvianConfig {
    /// Gravity vector
    pub gravity: Vector3<f32>,
    /// Number of substeps per physics tick
    pub substeps: u32,
    /// Enable CCD (Continuous Collision Detection)
    pub enable_ccd: bool,
    /// Enable sleeping for inactive bodies
    pub enable_sleeping: bool,
    /// Linear damping coefficient
    pub linear_damping: f32,
    /// Angular damping coefficient
    pub angular_damping: f32,
    /// Default friction coefficient
    pub default_friction: f32,
    /// Default restitution (bounciness)
    pub default_restitution: f32,
}

impl Default for AvianConfig {
    fn default() -> Self {
        Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            substeps: 4,
            enable_ccd: false,
            enable_sleeping: true,
            linear_damping: 0.0,
            angular_damping: 0.0,
            default_friction: 0.5,
            default_restitution: 0.3,
        }
    }
}

/// Avian body handle wrapping a Bevy Entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AvianBodyHandle(pub u64);

impl From<AvianBodyHandle> for u64 {
    fn from(handle: AvianBodyHandle) -> u64 {
        handle.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AvianColliderHandle(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AvianConstraintHandle(pub u64);

/// Internal body data for standalone mode
#[derive(Debug, Clone)]
struct BodyData {
    body_type: BodyType,
    transform: Transform,
    linear_velocity: Vector3<f32>,
    angular_velocity: Vector3<f32>,
    mass: f32,
    is_enabled: bool,
}

/// Internal collider data
#[derive(Debug, Clone)]
struct ColliderData {
    body_handle: AvianBodyHandle,
    shape_type: String,
    offset: Vector3<f32>,
    friction: f32,
    restitution: f32,
    is_sensor: bool,
    is_enabled: bool,
}

/// Avian physics backend with optional Bevy World integration
pub struct AvianBackend {
    config: AvianConfig,
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,

    // Body management (standalone mode)
    bodies: HashMap<u64, BodyData>,
    colliders: HashMap<u64, ColliderData>,
    next_id: u64,

    // Bevy integration (when feature enabled)
    #[cfg(feature = "avian3d")]
    world: Option<World>,
    #[cfg(feature = "avian3d")]
    entity_map: HashMap<u64, Entity>,

    initialized: bool,
    sim_time: f64,
}

impl AvianBackend {
    /// Create a new backend with optional Bevy World initialization
    #[cfg(feature = "avian3d")]
    fn init_bevy_world(&mut self) -> Result<(), BackendError> {
        let mut world = World::new();

        // Initialize required resources
        world.insert_resource(Time::<bevy_time::Fixed>::default());
        world.insert_resource(Gravity(bevy_math::Vec3::new(
            self.gravity.x,
            self.gravity.y,
            self.gravity.z,
        )));
        world.insert_resource(SubstepCount(self.config.substeps));

        self.world = Some(world);
        self.initialized = true;

        Ok(())
    }

    /// Get reference to internal Bevy World
    #[cfg(feature = "avian3d")]
    pub fn world(&self) -> Option<&World> {
        self.world.as_ref()
    }

    /// Get mutable reference to internal Bevy World
    #[cfg(feature = "avian3d")]
    pub fn world_mut(&mut self) -> Option<&mut World> {
        self.world.as_mut()
    }

    /// Spawn entity in Bevy World
    #[cfg(feature = "avian3d")]
    fn spawn_body(&mut self, desc: &BodyDesc) -> Result<Entity, BackendError> {
        let world = self.world.as_mut().ok_or_else(|| {
            BackendError::InitializationFailed("Bevy World not initialized".into())
        })?;

        let rigid_body = match desc.body_type {
            BodyType::Dynamic => RigidBody::Dynamic,
            BodyType::Static => RigidBody::Static,
            BodyType::Kinematic => RigidBody::Kinematic,
        };

        let transform = bevy_transform::components::Transform::from_xyz(
            desc.transform.position.x,
            desc.transform.position.y,
            desc.transform.position.z,
        ).with_rotation(bevy_math::Quat::from_xyzw(
            desc.transform.rotation.i,
            desc.transform.rotation.j,
            desc.transform.rotation.k,
            desc.transform.rotation.w,
        ));

        let linear_velocity = LinearVelocity(bevy_math::Vec3::new(
            desc.linear_velocity.x,
            desc.linear_velocity.y,
            desc.linear_velocity.z,
        ));

        let angular_velocity = AngularVelocity(bevy_math::Vec3::new(
            desc.angular_velocity.x,
            desc.angular_velocity.y,
            desc.angular_velocity.z,
        ));

        let entity = world.spawn((
            rigid_body,
            transform,
            linear_velocity,
            angular_velocity,
            LinearDamping(self.config.linear_damping),
            AngularDamping(self.config.angular_damping),
        )).id();

        Ok(entity)
    }

    /// Get Entity from handle
    #[cfg(feature = "avian3d")]
    fn get_entity(&self, handle: AvianBodyHandle) -> Option<Entity> {
        self.entity_map.get(&handle.0).copied()
    }

    /// Run physics step on Bevy World
    #[cfg(feature = "avian3d")]
    fn step_bevy(&mut self, dt: f32) -> Result<(), BackendError> {
        // In standalone mode, we simulate a simplified physics step
        // For full Bevy integration, this should be called from Bevy's schedule

        if let Some(ref mut world) = self.world {
            // Update time
            if let Some(mut time) = world.get_resource_mut::<Time<bevy_time::Fixed>>() {
                // Advance time - simplified
            }

            // Apply gravity to dynamic bodies
            let gravity = self.gravity;
            let mut query = world.query::<(&RigidBody, &mut LinearVelocity)>();

            for (rb, mut vel) in query.iter_mut(world) {
                if *rb == RigidBody::Dynamic {
                    vel.0.x += gravity.x * dt;
                    vel.0.y += gravity.y * dt;
                    vel.0.z += gravity.z * dt;
                }
            }

            // Update positions
            let mut pos_query = world.query::<(&RigidBody, &LinearVelocity, &mut bevy_transform::components::Transform)>();

            for (rb, vel, mut transform) in pos_query.iter_mut(world) {
                if *rb == RigidBody::Dynamic || *rb == RigidBody::Kinematic {
                    transform.translation.x += vel.0.x * dt;
                    transform.translation.y += vel.0.y * dt;
                    transform.translation.z += vel.0.z * dt;
                }
            }
        }

        Ok(())
    }

    /// Step physics in standalone mode (no Bevy)
    fn step_standalone(&mut self, dt: f32) {
        // Simple Euler integration for standalone mode
        for body in self.bodies.values_mut() {
            if body.body_type == BodyType::Dynamic && body.is_enabled {
                // Apply gravity
                body.linear_velocity += self.gravity * dt;

                // Apply damping
                body.linear_velocity *= 1.0 - self.config.linear_damping * dt;
                body.angular_velocity *= 1.0 - self.config.angular_damping * dt;

                // Integrate position
                body.transform.position.x += body.linear_velocity.x * dt;
                body.transform.position.y += body.linear_velocity.y * dt;
                body.transform.position.z += body.linear_velocity.z * dt;

                // Simple rotation integration (Euler)
                let omega = body.angular_velocity * dt;
                let dq = UnitQuaternion::from_scaled_axis(omega);
                body.transform.rotation = dq * body.transform.rotation;
            }
        }
    }

    /// Get all bodies with their transforms (for synchronization)
    pub fn get_body_states(&self) -> Vec<(AvianBodyHandle, Transform, Vector3<f32>, Vector3<f32>)> {
        self.bodies.iter().map(|(id, data)| {
            (AvianBodyHandle(*id), data.transform, data.linear_velocity, data.angular_velocity)
        }).collect()
    }

    /// Set body state from external source (for synchronization)
    pub fn set_body_state(
        &mut self,
        handle: AvianBodyHandle,
        transform: Transform,
        linear_velocity: Vector3<f32>,
        angular_velocity: Vector3<f32>,
    ) -> Result<(), BackendError> {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            body.transform = transform;
            body.linear_velocity = linear_velocity;
            body.angular_velocity = angular_velocity;
            Ok(())
        } else {
            Err(BackendError::InvalidHandle("Body not found".into()))
        }
    }

    /// Sync from Bevy World to internal state
    #[cfg(feature = "avian3d")]
    pub fn sync_from_bevy(&mut self) {
        if let Some(ref world) = self.world {
            for (id, entity) in &self.entity_map {
                if let Some(transform) = world.get::<bevy_transform::components::Transform>(*entity) {
                    if let Some(body) = self.bodies.get_mut(id) {
                        body.transform.position = Point3::new(
                            transform.translation.x,
                            transform.translation.y,
                            transform.translation.z,
                        );
                        body.transform.rotation = UnitQuaternion::from_quaternion(
                            nalgebra::Quaternion::new(
                                transform.rotation.w,
                                transform.rotation.x,
                                transform.rotation.y,
                                transform.rotation.z,
                            )
                        );
                    }
                }

                if let Some(vel) = world.get::<LinearVelocity>(*entity) {
                    if let Some(body) = self.bodies.get_mut(id) {
                        body.linear_velocity = Vector3::new(vel.0.x, vel.0.y, vel.0.z);
                    }
                }

                if let Some(ang_vel) = world.get::<AngularVelocity>(*entity) {
                    if let Some(body) = self.bodies.get_mut(id) {
                        body.angular_velocity = Vector3::new(ang_vel.0.x, ang_vel.0.y, ang_vel.0.z);
                    }
                }
            }
        }
    }

    /// Sync internal state to Bevy World
    #[cfg(feature = "avian3d")]
    pub fn sync_to_bevy(&mut self) {
        if let Some(ref mut world) = self.world {
            for (id, entity) in &self.entity_map {
                if let Some(body) = self.bodies.get(id) {
                    if let Some(mut transform) = world.get_mut::<bevy_transform::components::Transform>(*entity) {
                        transform.translation.x = body.transform.position.x;
                        transform.translation.y = body.transform.position.y;
                        transform.translation.z = body.transform.position.z;
                        transform.rotation = bevy_math::Quat::from_xyzw(
                            body.transform.rotation.i,
                            body.transform.rotation.j,
                            body.transform.rotation.k,
                            body.transform.rotation.w,
                        );
                    }

                    if let Some(mut vel) = world.get_mut::<LinearVelocity>(*entity) {
                        vel.0.x = body.linear_velocity.x;
                        vel.0.y = body.linear_velocity.y;
                        vel.0.z = body.linear_velocity.z;
                    }

                    if let Some(mut ang_vel) = world.get_mut::<AngularVelocity>(*entity) {
                        ang_vel.0.x = body.angular_velocity.x;
                        ang_vel.0.y = body.angular_velocity.y;
                        ang_vel.0.z = body.angular_velocity.z;
                    }
                }
            }
        }
    }
}

impl PhysicsBackend for AvianBackend {
    type Config = AvianConfig;
    type BodyHandle = AvianBodyHandle;
    type ColliderHandle = AvianColliderHandle;
    type ConstraintHandle = AvianConstraintHandle;

    fn new(config: Self::Config) -> Result<Self, BackendError> {
        let gravity = config.gravity;
        let mut backend = Self {
            config,
            gravity,
            contacts: Vec::new(),
            stats: SimulationStats::default(),
            bodies: HashMap::new(),
            colliders: HashMap::new(),
            next_id: 0,
            #[cfg(feature = "avian3d")]
            world: None,
            #[cfg(feature = "avian3d")]
            entity_map: HashMap::new(),
            initialized: false,
            sim_time: 0.0,
        };

        #[cfg(feature = "avian3d")]
        {
            backend.init_bevy_world()?;
        }

        backend.initialized = true;
        Ok(backend)
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "Avian",
            version: "0.4",
            description: "ECS-native Bevy physics engine",
            gpu_accelerated: false,
            differentiable: false,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            physics_3d: true,
            physics_2d: cfg!(feature = "avian2d"),
            soft_bodies: false,
            cloth: false,
            fluids: false,
            articulated: true,
            ccd: self.config.enable_ccd,
            deterministic: true,
            parallel: true,
            gpu: false,
            differentiable: false,
            max_bodies: 0, // No hard limit
        }
    }

    fn step(&mut self, dt: f32) {
        let start = std::time::Instant::now();

        #[cfg(feature = "avian3d")]
        {
            let _ = self.step_bevy(dt);
            self.sync_from_bevy();
        }

        #[cfg(not(feature = "avian3d"))]
        {
            self.step_standalone(dt);
        }

        self.sim_time += dt as f64;
        self.stats.total_us = start.elapsed().as_micros() as u64;
        self.stats.active_bodies = self.bodies.len() as u32;
    }

    fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.gravity = gravity;

        #[cfg(feature = "avian3d")]
        {
            if let Some(ref mut world) = self.world {
                world.insert_resource(Gravity(bevy_math::Vec3::new(gravity.x, gravity.y, gravity.z)));
            }
        }
    }

    fn gravity(&self) -> Vector3<f32> {
        self.gravity
    }

    fn create_body(&mut self, desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> {
        let id = self.next_id;
        self.next_id += 1;
        let handle = AvianBodyHandle(id);

        // Create body data
        let body_data = BodyData {
            body_type: desc.body_type.clone(),
            transform: desc.transform,
            linear_velocity: desc.linear_velocity,
            angular_velocity: desc.angular_velocity,
            mass: 1.0, // Default mass
            is_enabled: true,
        };

        self.bodies.insert(id, body_data);

        // Spawn in Bevy World if available
        #[cfg(feature = "avian3d")]
        {
            if let Ok(entity) = self.spawn_body(desc) {
                self.entity_map.insert(id, entity);
            }
        }

        Ok(handle)
    }

    fn remove_body(&mut self, handle: Self::BodyHandle) -> Result<(), BackendError> {
        self.bodies.remove(&handle.0);

        #[cfg(feature = "avian3d")]
        {
            if let Some(entity) = self.entity_map.remove(&handle.0) {
                if let Some(ref mut world) = self.world {
                    world.despawn(entity);
                }
            }
        }

        Ok(())
    }

    fn body_transform(&self, handle: Self::BodyHandle) -> Option<Transform> {
        self.bodies.get(&handle.0).map(|b| b.transform)
    }

    fn set_body_transform(&mut self, handle: Self::BodyHandle, transform: Transform) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            body.transform = transform;
        }

        #[cfg(feature = "avian3d")]
        {
            self.sync_to_bevy();
        }
    }

    fn body_linear_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        self.bodies.get(&handle.0).map(|b| b.linear_velocity)
    }

    fn set_body_linear_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            body.linear_velocity = velocity;
        }

        #[cfg(feature = "avian3d")]
        {
            self.sync_to_bevy();
        }
    }

    fn body_angular_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        self.bodies.get(&handle.0).map(|b| b.angular_velocity)
    }

    fn set_body_angular_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            body.angular_velocity = velocity;
        }

        #[cfg(feature = "avian3d")]
        {
            self.sync_to_bevy();
        }
    }

    fn apply_force(&mut self, handle: Self::BodyHandle, force: Vector3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            if body.body_type == BodyType::Dynamic {
                // F = ma, assume mass = 1 for simplicity
                body.linear_velocity += force / body.mass;
            }
        }
    }

    fn apply_force_at_point(&mut self, handle: Self::BodyHandle, force: Vector3<f32>, point: Point3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            if body.body_type == BodyType::Dynamic {
                // Apply linear force
                body.linear_velocity += force / body.mass;

                // Apply torque from offset
                let r = point - body.transform.position;
                let torque = r.cross(&force);
                body.angular_velocity += torque / body.mass; // Simplified inertia
            }
        }
    }

    fn apply_impulse(&mut self, handle: Self::BodyHandle, impulse: Vector3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            if body.body_type == BodyType::Dynamic {
                body.linear_velocity += impulse / body.mass;
            }
        }
    }

    fn apply_torque(&mut self, handle: Self::BodyHandle, torque: Vector3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            if body.body_type == BodyType::Dynamic {
                body.angular_velocity += torque / body.mass; // Simplified inertia
            }
        }
    }

    fn body_count(&self) -> usize {
        self.bodies.len()
    }

    fn create_collider(&mut self, body: Self::BodyHandle, desc: &ColliderDesc) -> Result<Self::ColliderHandle, BackendError> {
        let id = self.next_id;
        self.next_id += 1;
        let handle = AvianColliderHandle(id);

        let collider_data = ColliderData {
            body_handle: body,
            shape_type: format!("{:?}", desc.shape),
            offset: Vector3::zeros(),
            friction: self.config.default_friction,
            restitution: self.config.default_restitution,
            is_sensor: desc.is_sensor,
            is_enabled: true,
        };

        self.colliders.insert(id, collider_data);

        #[cfg(feature = "avian3d")]
        {
            // Add collider component to entity
            if let (Some(entity), Some(ref mut world)) = (self.entity_map.get(&body.0), self.world.as_mut()) {
                // Create Avian collider based on shape
                // This is simplified - full implementation would handle all shape types
                let collider = match &desc.shape {
                    crate::collider::Shape::Sphere { radius } => Collider::sphere(*radius),
                    crate::collider::Shape::Box { half_extents } => {
                        Collider::cuboid(half_extents.x, half_extents.y, half_extents.z)
                    }
                    crate::collider::Shape::Capsule { radius, height } => {
                        Collider::capsule(*height, *radius)
                    }
                    _ => Collider::sphere(0.5), // Fallback
                };

                world.entity_mut(*entity).insert((
                    collider,
                    Friction::new(self.config.default_friction),
                    Restitution::new(self.config.default_restitution),
                ));

                if desc.is_sensor {
                    world.entity_mut(*entity).insert(Sensor);
                }
            }
        }

        Ok(handle)
    }

    fn remove_collider(&mut self, handle: Self::ColliderHandle) -> Result<(), BackendError> {
        self.colliders.remove(&handle.0);
        Ok(())
    }

    fn set_collider_material(&mut self, handle: Self::ColliderHandle, material: PhysicsMaterial) {
        if let Some(collider) = self.colliders.get_mut(&handle.0) {
            collider.friction = material.friction;
            collider.restitution = material.restitution;
        }
    }

    fn set_collider_enabled(&mut self, handle: Self::ColliderHandle, enabled: bool) {
        if let Some(collider) = self.colliders.get_mut(&handle.0) {
            collider.is_enabled = enabled;
        }
    }

    fn collider_aabb(&self, _handle: Self::ColliderHandle) -> Option<AABB> {
        // Would need to compute from shape and transform
        None
    }

    fn create_constraint(&mut self, desc: &ConstraintDesc) -> Result<Self::ConstraintHandle, BackendError> {
        let id = self.next_id;
        self.next_id += 1;
        let handle = AvianConstraintHandle(id);

        #[cfg(feature = "avian3d")]
        {
            // Create Avian joint based on constraint type
            use crate::constraint::ConstraintType;

            if let (Some(ref mut world), Some(entity_a), Some(entity_b)) = (
                self.world.as_mut(),
                self.entity_map.get(&desc.body_a.0),
                self.entity_map.get(&desc.body_b.0),
            ) {
                match desc.constraint_type {
                    ConstraintType::Fixed => {
                        world.entity_mut(*entity_a).insert(FixedJoint::new(*entity_a, *entity_b));
                    }
                    ConstraintType::Distance { distance } => {
                        let mut joint = DistanceJoint::new(*entity_a, *entity_b);
                        joint.rest_length = distance;
                        world.entity_mut(*entity_a).insert(joint);
                    }
                    ConstraintType::Hinge { axis, .. } => {
                        let joint = RevoluteJoint::new(*entity_a, *entity_b)
                            .with_aligned_axis(bevy_math::Vec3::new(axis.x, axis.y, axis.z));
                        world.entity_mut(*entity_a).insert(joint);
                    }
                    ConstraintType::Slider { axis, .. } => {
                        let joint = PrismaticJoint::new(*entity_a, *entity_b)
                            .with_free_axis(bevy_math::Vec3::new(axis.x, axis.y, axis.z));
                        world.entity_mut(*entity_a).insert(joint);
                    }
                    _ => {
                        return Err(BackendError::Unsupported(format!("Constraint type not supported")));
                    }
                }
            }
        }

        Ok(handle)
    }

    fn remove_constraint(&mut self, _handle: Self::ConstraintHandle) -> Result<(), BackendError> {
        // Would need to track and remove joint components
        Ok(())
    }

    fn ray_cast(&self, ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> {
        #[cfg(feature = "avian3d")]
        {
            // Would use Avian's SpatialQuery
            // This requires access to the physics world's spatial data structures
        }
        None
    }

    fn ray_cast_all(&self, _ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>> {
        Vec::new()
    }

    fn shape_cast(&self, _cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>> {
        None
    }

    fn query_aabb(&self, _aabb: &AABB) -> Vec<Self::BodyHandle> {
        Vec::new()
    }

    fn contacts(&self) -> &[ContactManifold] {
        &self.contacts
    }

    fn serialize_state(&self) -> Result<Vec<u8>, BackendError> {
        let state: Vec<(u64, Transform, Vector3<f32>, Vector3<f32>)> = self.bodies.iter()
            .map(|(id, body)| (*id, body.transform, body.linear_velocity, body.angular_velocity))
            .collect();

        bincode::serialize(&state)
            .map_err(|e| BackendError::SerializationError(e.to_string()))
    }

    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), BackendError> {
        let state: Vec<(u64, Transform, Vector3<f32>, Vector3<f32>)> = bincode::deserialize(data)
            .map_err(|e| BackendError::DeserializationError(e.to_string()))?;

        for (id, transform, linear_vel, angular_vel) in state {
            if let Some(body) = self.bodies.get_mut(&id) {
                body.transform = transform;
                body.linear_velocity = linear_vel;
                body.angular_velocity = angular_vel;
            }
        }

        #[cfg(feature = "avian3d")]
        {
            self.sync_to_bevy();
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.bodies.clear();
        self.colliders.clear();
        self.contacts.clear();
        self.sim_time = 0.0;

        #[cfg(feature = "avian3d")]
        {
            self.entity_map.clear();
            if let Some(ref mut world) = self.world {
                world.clear_all();
            }
            let _ = self.init_bevy_world();
        }
    }

    fn stats(&self) -> SimulationStats {
        self.stats.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Bridge for connecting HyperPhysics with an external Bevy World
#[cfg(feature = "avian3d")]
pub struct AvianBridge<'w> {
    world: &'w mut World,
}

#[cfg(feature = "avian3d")]
impl<'w> AvianBridge<'w> {
    /// Create a bridge from a mutable World reference
    pub fn from_world(world: &'w mut World) -> Self {
        Self { world }
    }

    /// Get body count from Bevy World
    pub fn body_count(&self) -> usize {
        self.world.query::<&RigidBody>().iter(self.world).count()
    }

    /// Get all rigid bodies with their transforms
    pub fn get_bodies(&self) -> Vec<(Entity, Transform)> {
        let mut result = Vec::new();
        let mut query = self.world.query::<(Entity, &RigidBody, &bevy_transform::components::Transform)>();

        for (entity, _rb, transform) in query.iter(self.world) {
            let t = Transform {
                position: Point3::new(
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ),
                rotation: UnitQuaternion::from_quaternion(
                    nalgebra::Quaternion::new(
                        transform.rotation.w,
                        transform.rotation.x,
                        transform.rotation.y,
                        transform.rotation.z,
                    )
                ),
            };
            result.push((entity, t));
        }

        result
    }

    /// Apply force to an entity
    pub fn apply_force(&mut self, entity: Entity, force: Vector3<f32>) {
        if let Some(mut vel) = self.world.get_mut::<LinearVelocity>(entity) {
            // Simplified - would need mass for proper force application
            vel.0.x += force.x;
            vel.0.y += force.y;
            vel.0.z += force.z;
        }
    }

    /// Get gravity
    pub fn gravity(&self) -> Vector3<f32> {
        self.world.get_resource::<Gravity>()
            .map(|g| Vector3::new(g.0.x, g.0.y, g.0.z))
            .unwrap_or(Vector3::new(0.0, -9.81, 0.0))
    }

    /// Set gravity
    pub fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.world.insert_resource(Gravity(bevy_math::Vec3::new(gravity.x, gravity.y, gravity.z)));
    }
}
