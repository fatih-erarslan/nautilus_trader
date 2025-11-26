//! MuJoCo physics integration for HyperPhysics
//!
//! Google DeepMind's MuJoCo (Multi-Joint dynamics with Contact) is a physics
//! engine optimized for robotics, biomechanics, and machine learning research.
//!
//! ## Features
//! - Contact-rich simulation
//! - Articulated body dynamics
//! - Muscle/tendon modeling
//! - Differentiable physics (via autodiff)
//! - XML model format (MJCF)

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use nalgebra::Vector3;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

// Include generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Error type for MuJoCo operations
#[derive(Debug, thiserror::Error)]
pub enum MuJoCoError {
    #[error("Failed to load model: {0}")]
    ModelLoadFailed(String),
    #[error("Failed to create simulation data")]
    DataCreationFailed,
    #[error("Model not loaded")]
    NoModel,
    #[error("Invalid path: {0}")]
    InvalidPath(String),
    #[error("Simulation error: {0}")]
    SimulationError(String),
}

pub type Result<T> = std::result::Result<T, MuJoCoError>;

/// MuJoCo configuration
#[derive(Debug, Clone)]
pub struct MuJoCoConfig {
    pub timestep: f64,
    pub gravity: Vector3<f64>,
    pub enable_contact: bool,
    pub solver_iterations: i32,
}

impl Default for MuJoCoConfig {
    fn default() -> Self {
        Self {
            timestep: 0.002, // 500 Hz
            gravity: Vector3::new(0.0, 0.0, -9.81),
            enable_contact: true,
            solver_iterations: 100,
        }
    }
}

/// Safe wrapper around MuJoCo model and data
pub struct MuJoCoAdapter {
    model: *mut mjModel,
    data: *mut mjData,
    config: MuJoCoConfig,
}

// Safety: MuJoCo is internally thread-safe for read operations
// Write operations should be synchronized externally
unsafe impl Send for MuJoCoAdapter {}

impl MuJoCoAdapter {
    /// Create a new MuJoCo adapter (without loading a model)
    pub fn new(config: MuJoCoConfig) -> Self {
        Self {
            model: ptr::null_mut(),
            data: ptr::null_mut(),
            config,
        }
    }

    /// Load a model from an MJCF XML file
    pub fn load_xml<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path_str = path.as_ref().to_str()
            .ok_or_else(|| MuJoCoError::InvalidPath("Invalid UTF-8 in path".into()))?;

        let c_path = CString::new(path_str)
            .map_err(|_| MuJoCoError::InvalidPath("Path contains null byte".into()))?;

        // Error buffer
        let mut error_buf = [0i8; 1000];

        unsafe {
            // Free existing model if any
            self.free_model();

            // Load new model
            let model = mj_loadXML(
                c_path.as_ptr(),
                ptr::null(),
                error_buf.as_mut_ptr(),
                error_buf.len() as i32,
            );

            if model.is_null() {
                let error_msg = CStr::from_ptr(error_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                return Err(MuJoCoError::ModelLoadFailed(error_msg));
            }

            self.model = model;

            // Create simulation data
            let data = mj_makeData(self.model);
            if data.is_null() {
                mj_deleteModel(self.model);
                self.model = ptr::null_mut();
                return Err(MuJoCoError::DataCreationFailed);
            }

            self.data = data;
        }

        Ok(())
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self) -> bool {
        !self.model.is_null() && !self.data.is_null()
    }

    /// Step the simulation forward
    pub fn step(&mut self) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }

        unsafe {
            mj_step(self.model, self.data);
        }

        Ok(())
    }

    /// Step forward kinematics only (no dynamics)
    pub fn forward(&mut self) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }

        unsafe {
            mj_forward(self.model, self.data);
        }

        Ok(())
    }

    /// Reset simulation to initial state
    pub fn reset(&mut self) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }

        unsafe {
            mj_resetData(self.model, self.data);
        }

        Ok(())
    }

    /// Get current simulation time
    pub fn time(&self) -> f64 {
        if !self.is_loaded() {
            return 0.0;
        }
        unsafe { (*self.data).time }
    }

    /// Get number of bodies in the model
    pub fn body_count(&self) -> i32 {
        if !self.is_loaded() {
            return 0;
        }
        unsafe { (*self.model).nbody }
    }

    /// Get number of joints in the model
    pub fn joint_count(&self) -> i32 {
        if !self.is_loaded() {
            return 0;
        }
        unsafe { (*self.model).njnt }
    }

    /// Get number of actuators in the model
    pub fn actuator_count(&self) -> i32 {
        if !self.is_loaded() {
            return 0;
        }
        unsafe { (*self.model).nu }
    }

    /// Get number of generalized coordinates
    pub fn nq(&self) -> i32 {
        if !self.is_loaded() {
            return 0;
        }
        unsafe { (*self.model).nq }
    }

    /// Get number of degrees of freedom
    pub fn nv(&self) -> i32 {
        if !self.is_loaded() {
            return 0;
        }
        unsafe { (*self.model).nv }
    }

    /// Get generalized positions (qpos)
    pub fn get_qpos(&self) -> Vec<f64> {
        if !self.is_loaded() {
            return Vec::new();
        }
        unsafe {
            let nq = (*self.model).nq as usize;
            let qpos_ptr = (*self.data).qpos;
            if qpos_ptr.is_null() {
                return Vec::new();
            }
            std::slice::from_raw_parts(qpos_ptr, nq).to_vec()
        }
    }

    /// Set generalized positions (qpos)
    pub fn set_qpos(&mut self, qpos: &[f64]) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }
        unsafe {
            let nq = (*self.model).nq as usize;
            let qpos_ptr = (*self.data).qpos;
            if qpos_ptr.is_null() || qpos.len() != nq {
                return Err(MuJoCoError::SimulationError("Invalid qpos size".into()));
            }
            std::ptr::copy_nonoverlapping(qpos.as_ptr(), qpos_ptr, nq);
        }
        Ok(())
    }

    /// Get generalized velocities (qvel)
    pub fn get_qvel(&self) -> Vec<f64> {
        if !self.is_loaded() {
            return Vec::new();
        }
        unsafe {
            let nv = (*self.model).nv as usize;
            let qvel_ptr = (*self.data).qvel;
            if qvel_ptr.is_null() {
                return Vec::new();
            }
            std::slice::from_raw_parts(qvel_ptr, nv).to_vec()
        }
    }

    /// Set generalized velocities (qvel)
    pub fn set_qvel(&mut self, qvel: &[f64]) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }
        unsafe {
            let nv = (*self.model).nv as usize;
            let qvel_ptr = (*self.data).qvel;
            if qvel_ptr.is_null() || qvel.len() != nv {
                return Err(MuJoCoError::SimulationError("Invalid qvel size".into()));
            }
            std::ptr::copy_nonoverlapping(qvel.as_ptr(), qvel_ptr, nv);
        }
        Ok(())
    }

    /// Get actuator control signals
    pub fn get_ctrl(&self) -> Vec<f64> {
        if !self.is_loaded() {
            return Vec::new();
        }
        unsafe {
            let nu = (*self.model).nu as usize;
            let ctrl_ptr = (*self.data).ctrl;
            if ctrl_ptr.is_null() || nu == 0 {
                return Vec::new();
            }
            std::slice::from_raw_parts(ctrl_ptr, nu).to_vec()
        }
    }

    /// Set actuator control signals
    pub fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }
        unsafe {
            let nu = (*self.model).nu as usize;
            let ctrl_ptr = (*self.data).ctrl;
            if ctrl_ptr.is_null() || ctrl.len() != nu {
                return Err(MuJoCoError::SimulationError("Invalid ctrl size".into()));
            }
            std::ptr::copy_nonoverlapping(ctrl.as_ptr(), ctrl_ptr, nu);
        }
        Ok(())
    }

    /// Get MuJoCo library version
    pub fn version() -> i32 {
        unsafe { mj_version() }
    }

    /// Free the current model and data
    fn free_model(&mut self) {
        unsafe {
            if !self.data.is_null() {
                mj_deleteData(self.data);
                self.data = ptr::null_mut();
            }
            if !self.model.is_null() {
                mj_deleteModel(self.model);
                self.model = ptr::null_mut();
            }
        }
    }
}

impl Drop for MuJoCoAdapter {
    fn drop(&mut self) {
        self.free_model();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = MuJoCoAdapter::new(MuJoCoConfig::default());
        assert!(!adapter.is_loaded());
    }

    #[test]
    fn test_version() {
        // This will return 0 in mock mode
        let version = MuJoCoAdapter::version();
        println!("MuJoCo version: {}", version);
    }
}
