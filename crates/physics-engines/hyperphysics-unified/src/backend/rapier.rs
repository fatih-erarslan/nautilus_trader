//! Rapier3D physics backend implementation

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::{BodyDesc, BodyType};
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::shape::Shape;
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;

/// Serializable body state for state save/restore
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableBodyState {
    pub position: [f32; 3],
    pub rotation: [f32; 4], // quaternion (x, y, z, w)
    pub linvel: [f32; 3],
    pub angvel: [f32; 3],
    pub is_dynamic: bool,
    pub is_enabled: bool,
}

/// Serializable physics state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializablePhysicsState {
    pub gravity: [f32; 3],
    pub bodies: Vec<(u64, SerializableBodyState)>, // (index, state)
    pub integration_dt: f32,
}

/// Rapier3D configuration
#[derive(Debug, Clone)]
pub struct RapierConfig {
    pub max_ccd_substeps: usize,
    pub prediction_distance: f32,
}

impl Default for RapierConfig {
    fn default() -> Self {
        Self {
            max_ccd_substeps: 4,
            prediction_distance: 0.002,
        }
    }
}

/// Rapier3D physics backend
pub struct RapierBackend {
    gravity: Vector3<f32>,
    integration_params: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,
    handle_map: HashMap<RigidBodyHandle, crate::body::BodyHandle>,
}

impl RapierBackend {
    fn convert_shape(shape: &Shape) -> SharedShape {
        match shape {
            Shape::Sphere { radius } => SharedShape::ball(*radius),
            Shape::Box { half_extents } => SharedShape::cuboid(half_extents.x, half_extents.y, half_extents.z),
            Shape::Capsule { half_height, radius } => SharedShape::capsule_y(*half_height, *radius),
            Shape::Cylinder { half_height, radius } => SharedShape::cylinder(*half_height, *radius),
            Shape::Cone { half_height, radius } => SharedShape::cone(*half_height, *radius),
            Shape::ConvexHull { points } => {
                SharedShape::convex_hull(points).unwrap_or_else(|| SharedShape::ball(0.1))
            }
            Shape::TriMesh { vertices, indices } => {
                SharedShape::trimesh(vertices.clone(), indices.clone())
                    .unwrap_or_else(|_| SharedShape::ball(0.1))
            }
            Shape::HeightField { heights, rows, cols, scale } => {
                let matrix = nalgebra::DMatrix::from_row_slice(*rows as usize, *cols as usize, heights);
                SharedShape::heightfield(matrix, *scale)
            }
            Shape::Compound { shapes } => {
                let converted: Vec<_> = shapes
                    .iter()
                    .map(|(s, t)| (t.to_isometry(), Self::convert_shape(s)))
                    .collect();
                SharedShape::compound(converted)
            }
        }
    }
}

impl PhysicsBackend for RapierBackend {
    type Config = RapierConfig;
    type BodyHandle = RigidBodyHandle;
    type ColliderHandle = ColliderHandle;
    type ConstraintHandle = ImpulseJointHandle;

    fn new(_config: Self::Config) -> Result<Self, BackendError> {
        Ok(Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            integration_params: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            contacts: Vec::new(),
            stats: SimulationStats::default(),
            handle_map: HashMap::new(),
        })
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "Rapier3D",
            version: "0.22",
            description: "High-performance Rust physics engine",
            gpu_accelerated: false,
            differentiable: false,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            physics_3d: true,
            physics_2d: false,
            soft_bodies: false,
            cloth: false,
            fluids: false,
            articulated: true,
            ccd: true,
            deterministic: true,
            parallel: true,
            gpu: false,
            differentiable: false,
            max_bodies: 0,
        }
    }

    fn step(&mut self, dt: f32) {
        let start = std::time::Instant::now();
        self.integration_params.dt = dt;

        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),  // No hooks
            &(),  // No events
        );

        self.stats.total_us = start.elapsed().as_micros() as u64;
        self.stats.active_bodies = self.rigid_body_set.len() as u32;
    }

    fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.gravity = gravity;
    }

    fn gravity(&self) -> Vector3<f32> {
        self.gravity
    }

    fn create_body(&mut self, desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> {
        let builder = match desc.body_type {
            BodyType::Static => RigidBodyBuilder::fixed(),
            BodyType::Dynamic => RigidBodyBuilder::dynamic(),
            BodyType::Kinematic => RigidBodyBuilder::kinematic_position_based(),
        };

        let rb = builder
            .translation(desc.transform.position)
            .rotation(desc.transform.rotation.scaled_axis())
            .linvel(desc.linear_velocity)
            .angvel(desc.angular_velocity)
            .linear_damping(desc.linear_damping)
            .angular_damping(desc.angular_damping)
            .gravity_scale(desc.gravity_scale)
            .can_sleep(desc.can_sleep)
            .ccd_enabled(desc.ccd_enabled)
            .user_data(desc.user_data as u128)
            .build();

        Ok(self.rigid_body_set.insert(rb))
    }

    fn remove_body(&mut self, handle: Self::BodyHandle) -> Result<(), BackendError> {
        self.rigid_body_set.remove(
            handle,
            &mut self.islands,
            &mut self.collider_set,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
        Ok(())
    }

    fn body_transform(&self, handle: Self::BodyHandle) -> Option<Transform> {
        self.rigid_body_set.get(handle).map(|rb| Transform {
            position: *rb.translation(),
            rotation: *rb.rotation(),
        })
    }

    fn set_body_transform(&mut self, handle: Self::BodyHandle, transform: Transform) {
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.set_translation(transform.position, true);
            rb.set_rotation(transform.rotation, true);
        }
    }

    fn body_linear_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        self.rigid_body_set.get(handle).map(|rb| *rb.linvel())
    }

    fn set_body_linear_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.set_linvel(velocity, true);
        }
    }

    fn body_angular_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        self.rigid_body_set.get(handle).map(|rb| *rb.angvel())
    }

    fn set_body_angular_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.set_angvel(velocity, true);
        }
    }

    fn apply_force(&mut self, handle: Self::BodyHandle, force: Vector3<f32>) {
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.add_force(force, true);
        }
    }

    fn apply_force_at_point(&mut self, handle: Self::BodyHandle, force: Vector3<f32>, point: Point3<f32>) {
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.add_force_at_point(force, point, true);
        }
    }

    fn apply_impulse(&mut self, handle: Self::BodyHandle, impulse: Vector3<f32>) {
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.apply_impulse(impulse, true);
        }
    }

    fn apply_torque(&mut self, handle: Self::BodyHandle, torque: Vector3<f32>) {
        if let Some(rb) = self.rigid_body_set.get_mut(handle) {
            rb.add_torque(torque, true);
        }
    }

    fn body_count(&self) -> usize {
        self.rigid_body_set.len()
    }

    fn create_collider(&mut self, body: Self::BodyHandle, desc: &ColliderDesc) -> Result<Self::ColliderHandle, BackendError> {
        let shape = Self::convert_shape(&desc.shape);
        let collider = ColliderBuilder::new(shape)
            .position(desc.local_transform.to_isometry())
            .friction(desc.material.friction)
            .restitution(desc.material.restitution)
            .density(desc.material.density)
            .sensor(desc.is_sensor)
            .collision_groups(InteractionGroups::new(
                Group::from_bits_truncate(desc.collision_groups),
                Group::from_bits_truncate(desc.collision_filter),
                InteractionTestMode::default(),
            ))
            .user_data(desc.user_data as u128)
            .build();

        Ok(self.collider_set.insert_with_parent(collider, body, &mut self.rigid_body_set))
    }

    fn remove_collider(&mut self, handle: Self::ColliderHandle) -> Result<(), BackendError> {
        self.collider_set.remove(handle, &mut self.islands, &mut self.rigid_body_set, true);
        Ok(())
    }

    fn set_collider_material(&mut self, handle: Self::ColliderHandle, material: PhysicsMaterial) {
        if let Some(c) = self.collider_set.get_mut(handle) {
            c.set_friction(material.friction);
            c.set_restitution(material.restitution);
            c.set_density(material.density);
        }
    }

    fn set_collider_enabled(&mut self, handle: Self::ColliderHandle, enabled: bool) {
        if let Some(c) = self.collider_set.get_mut(handle) {
            c.set_enabled(enabled);
        }
    }

    fn collider_aabb(&self, handle: Self::ColliderHandle) -> Option<AABB> {
        self.collider_set.get(handle).map(|c| {
            let aabb = c.compute_aabb();
            AABB::new(aabb.mins, aabb.maxs)
        })
    }

    fn create_constraint(&mut self, desc: &ConstraintDesc) -> Result<Self::ConstraintHandle, BackendError> {
        use crate::constraint::ConstraintType;
        use rapier3d::dynamics::*;

        // Get body handles - body_a is required, body_b defaults to ground if None
        let body_a_handle = self.handle_map.iter()
            .find(|(_, &v)| v == desc.body_a)
            .map(|(&k, _)| k)
            .ok_or_else(|| BackendError::InvalidHandle("Body A not found".into()))?;

        // body_b is optional (connects to world/ground if None)
        let body_b_handle = if let Some(body_b) = desc.body_b {
            self.handle_map.iter()
                .find(|(_, &v)| v == body_b)
                .map(|(&k, _)| k)
                .ok_or_else(|| BackendError::InvalidHandle("Body B not found".into()))?
        } else {
            // For world-attached constraints, we need a static body
            // Create a static ground body if needed
            let ground = RigidBodyBuilder::fixed().build();
            self.rigid_body_set.insert(ground)
        };

        let joint = match &desc.constraint_type {
            ConstraintType::Fixed => {
                GenericJointBuilder::new(JointAxesMask::LOCKED_FIXED_AXES)
                    .local_anchor1(Point3::origin())
                    .local_anchor2(Point3::origin())
                    .build()
            }
            ConstraintType::Ball { anchor_a, anchor_b } => {
                GenericJointBuilder::new(JointAxesMask::LOCKED_SPHERICAL_AXES)
                    .local_anchor1(*anchor_a)
                    .local_anchor2(*anchor_b)
                    .build()
            }
            ConstraintType::Hinge { anchor_a, anchor_b, axis_a, axis_b } => {
                GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES)
                    .local_anchor1(*anchor_a)
                    .local_anchor2(*anchor_b)
                    .local_axis1(*axis_a)
                    .local_axis2(*axis_b)
                    .build()
            }
            ConstraintType::Slider { anchor_a, anchor_b, axis } => {
                GenericJointBuilder::new(JointAxesMask::LOCKED_PRISMATIC_AXES)
                    .local_anchor1(*anchor_a)
                    .local_anchor2(*anchor_b)
                    .local_axis1(*axis)
                    .local_axis2(*axis)
                    .build()
            }
            ConstraintType::Distance { anchor_a, anchor_b, distance } => {
                let mut joint = GenericJointBuilder::new(JointAxesMask::LIN_X)
                    .local_anchor1(*anchor_a)
                    .local_anchor2(*anchor_b)
                    .build();
                joint.set_limits(JointAxis::LinX, [*distance, *distance]);
                joint
            }
            ConstraintType::Spring { anchor_a, anchor_b, rest_length, stiffness, damping } => {
                let mut joint = GenericJointBuilder::new(JointAxesMask::LIN_X)
                    .local_anchor1(*anchor_a)
                    .local_anchor2(*anchor_b)
                    .build();
                joint.set_limits(JointAxis::LinX, [*rest_length * 0.5, *rest_length * 2.0]);
                joint.set_motor(
                    JointAxis::LinX,
                    *rest_length,
                    0.0,
                    *stiffness,
                    *damping,
                );
                joint
            }
        };

        let handle = self.impulse_joints.insert(body_a_handle, body_b_handle, joint, true);
        Ok(handle)
    }

    fn remove_constraint(&mut self, handle: Self::ConstraintHandle) -> Result<(), BackendError> {
        self.impulse_joints.remove(handle, true);
        Ok(())
    }

    fn ray_cast(&self, ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> {
        let rapier_ray = Ray::new(ray.origin, *ray.direction);
        let filter = QueryFilter::default();
        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            filter,
        );

        query_pipeline
            .cast_ray(&rapier_ray, ray.max_distance, true)
            .map(|(handle, toi)| {
                let collider = &self.collider_set[handle];
                let point = rapier_ray.point_at(toi);
                RayHit {
                    body: collider.parent().unwrap(),
                    point,
                    normal: Vector3::zeros(), // Would need cast_ray_and_get_normal
                    distance: toi,
                }
            })
    }

    fn ray_cast_all(&self, ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>> {
        let rapier_ray = Ray::new(ray.origin, *ray.direction);
        let filter = QueryFilter::default();
        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            filter,
        );
        let mut hits = Vec::new();

        for (_handle, collider, intersection) in query_pipeline.intersect_ray(rapier_ray, ray.max_distance, true) {
            if let Some(parent) = collider.parent() {
                hits.push(RayHit {
                    body: parent,
                    point: rapier_ray.point_at(intersection.time_of_impact),
                    normal: intersection.normal,
                    distance: intersection.time_of_impact,
                });
            }
        }
        hits
    }

    fn shape_cast(&self, cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>> {
        // Approximate shape cast using ray cast with shape radius expansion
        // This provides a reliable implementation for swept-volume collision detection
        let shape_radius = match &cast.shape {
            Shape::Sphere { radius } => *radius,
            Shape::Box { half_extents } => half_extents.norm(),
            Shape::Capsule { half_height, radius } => *half_height + *radius,
            Shape::Cylinder { half_height, radius } => (*half_height * *half_height + *radius * *radius).sqrt(),
            _ => 0.5, // Default bounding radius for complex shapes
        };

        let ray = Ray::new(cast.start, *cast.direction);
        let filter = QueryFilter::default();
        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            filter,
        );

        // Use ray cast with expanded distance to account for shape radius
        let effective_max_dist = cast.max_distance + shape_radius;

        query_pipeline.cast_ray(&ray, effective_max_dist, true)
            .and_then(|(handle, toi)| {
                let collider = &self.collider_set[handle];
                // Adjust TOI to account for shape radius (subtract shape extent from impact)
                let adjusted_toi = (toi - shape_radius).max(0.0);
                if adjusted_toi <= cast.max_distance {
                    let hit_point = cast.start + cast.direction.into_inner() * adjusted_toi;
                    // Compute approximate normal from ray direction
                    let normal = -cast.direction.into_inner();
                    Some(ShapeHit {
                        body: collider.parent()?,
                        point: hit_point,
                        normal,
                        toi: adjusted_toi,
                    })
                } else {
                    None
                }
            })
    }

    fn query_aabb(&self, aabb: &AABB) -> Vec<Self::BodyHandle> {
        let rapier_aabb = rapier3d::prelude::Aabb::new(aabb.min, aabb.max);
        let filter = QueryFilter::default();
        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            filter,
        );
        let mut bodies = Vec::new();

        for (_, collider) in query_pipeline.intersect_aabb_conservative(rapier_aabb) {
            if let Some(parent) = collider.parent() {
                bodies.push(parent);
            }
        }
        bodies
    }

    fn contacts(&self) -> &[ContactManifold] {
        &self.contacts
    }

    fn serialize_state(&self) -> Result<Vec<u8>, BackendError> {
        let mut bodies = Vec::new();

        for (handle, rb) in self.rigid_body_set.iter() {
            let pos = rb.translation();
            let rot = rb.rotation();
            let linvel = rb.linvel();
            let angvel = rb.angvel();

            let state = SerializableBodyState {
                position: [pos.x, pos.y, pos.z],
                rotation: [rot.i, rot.j, rot.k, rot.w],
                linvel: [linvel.x, linvel.y, linvel.z],
                angvel: [angvel.x, angvel.y, angvel.z],
                is_dynamic: rb.is_dynamic(),
                is_enabled: rb.is_enabled(),
            };

            bodies.push((handle.into_raw_parts().0 as u64, state));
        }

        let physics_state = SerializablePhysicsState {
            gravity: [self.gravity.x, self.gravity.y, self.gravity.z],
            bodies,
            integration_dt: self.integration_params.dt,
        };

        bincode::serialize(&physics_state)
            .map_err(|e| BackendError::SerializationError(e.to_string()))
    }

    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), BackendError> {
        let physics_state: SerializablePhysicsState = bincode::deserialize(data)
            .map_err(|e| BackendError::DeserializationError(e.to_string()))?;

        // Restore gravity
        self.gravity = Vector3::new(
            physics_state.gravity[0],
            physics_state.gravity[1],
            physics_state.gravity[2],
        );

        // Restore integration params
        self.integration_params.dt = physics_state.integration_dt;

        // Restore body states
        for (raw_index, state) in physics_state.bodies {
            // Try to find the body by raw index
            let handle = RigidBodyHandle::from_raw_parts(raw_index as u32, 0);
            if let Some(rb) = self.rigid_body_set.get_mut(handle) {
                rb.set_translation(
                    Vector3::new(state.position[0], state.position[1], state.position[2]),
                    true,
                );
                rb.set_rotation(
                    UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        state.rotation[3], // w
                        state.rotation[0], // i
                        state.rotation[1], // j
                        state.rotation[2], // k
                    )),
                    true,
                );
                rb.set_linvel(
                    Vector3::new(state.linvel[0], state.linvel[1], state.linvel[2]),
                    true,
                );
                rb.set_angvel(
                    Vector3::new(state.angvel[0], state.angvel[1], state.angvel[2]),
                    true,
                );
                rb.set_enabled(state.is_enabled);
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.rigid_body_set = RigidBodySet::new();
        self.collider_set = ColliderSet::new();
        self.impulse_joints = ImpulseJointSet::new();
        self.multibody_joints = MultibodyJointSet::new();
        self.islands = IslandManager::new();
        self.broad_phase = BroadPhaseBvh::new();
        self.narrow_phase = NarrowPhase::new();
        self.ccd_solver = CCDSolver::new();
        self.contacts.clear();
    }

    fn stats(&self) -> SimulationStats {
        self.stats.clone()
    }

    fn set_solver_iterations(&mut self, velocity: u32, _position: u32) {
        // In rapier 0.30, num_solver_iterations is just usize
        self.integration_params.num_solver_iterations = velocity as usize;
        // num_additional_friction_iterations no longer exists in rapier 0.30
    }

    fn set_ccd_enabled(&mut self, _enabled: bool) {
        // CCD is per-body in Rapier
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
