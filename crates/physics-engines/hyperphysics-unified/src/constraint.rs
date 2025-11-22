//! Constraint/joint definitions

use crate::body::BodyHandle;
use nalgebra::{Point3, UnitVector3, Vector3};
use serde::{Deserialize, Serialize};
use slotmap::new_key_type;

new_key_type! {
    /// Handle to a constraint
    pub struct ConstraintHandle;
}

/// Constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Fixed joint - locks relative transform
    Fixed,
    /// Ball joint - 3 rotational DOF
    Ball { anchor_a: Point3<f32>, anchor_b: Point3<f32> },
    /// Hinge joint - 1 rotational DOF
    Hinge {
        anchor_a: Point3<f32>,
        anchor_b: Point3<f32>,
        axis_a: UnitVector3<f32>,
        axis_b: UnitVector3<f32>,
    },
    /// Slider joint - 1 translational DOF
    Slider {
        anchor_a: Point3<f32>,
        anchor_b: Point3<f32>,
        axis: UnitVector3<f32>,
    },
    /// Distance constraint
    Distance { anchor_a: Point3<f32>, anchor_b: Point3<f32>, distance: f32 },
    /// Spring constraint
    Spring {
        anchor_a: Point3<f32>,
        anchor_b: Point3<f32>,
        rest_length: f32,
        stiffness: f32,
        damping: f32,
    },
}

/// Constraint description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDesc {
    /// First body
    pub body_a: BodyHandle,
    /// Second body (None = world)
    pub body_b: Option<BodyHandle>,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Is enabled
    pub enabled: bool,
}

/// Runtime constraint state
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Constraint handle
    pub handle: ConstraintHandle,
    /// First body
    pub body_a: BodyHandle,
    /// Second body
    pub body_b: Option<BodyHandle>,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Is enabled
    pub enabled: bool,
    /// Accumulated impulse (for warm starting)
    pub impulse: Vector3<f32>,
}
