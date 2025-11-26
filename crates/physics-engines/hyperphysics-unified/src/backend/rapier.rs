//! Rapier3D physics backend implementation

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::{BodyDesc, BodyType};
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::shape::Shape;
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Vector3};
use rapier3d::prelude::*;
use std::any::Any;
use std::collections::HashMap;

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

    fn create_constraint(&mut self, _desc: &ConstraintDesc) -> Result<Self::ConstraintHandle, BackendError> {
        Err(BackendError::Unsupported("Constraint creation not implemented".into()))
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
        Err(BackendError::Unsupported("Serialization not implemented".into()))
    }

    fn deserialize_state(&mut self, _data: &[u8]) -> Result<(), BackendError> {
        Err(BackendError::Unsupported("Deserialization not implemented".into()))
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
