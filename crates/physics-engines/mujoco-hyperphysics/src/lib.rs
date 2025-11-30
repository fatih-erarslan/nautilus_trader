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
///
/// This struct provides a safe Rust interface to the MuJoCo physics engine.
/// All unsafe FFI calls are encapsulated with proper null checks and error handling.
///
/// # Thread Safety
///
/// MuJoCoAdapter implements Send but not Sync. This means:
/// - It can be moved between threads
/// - It cannot be shared between threads without external synchronization
/// - Read operations (get_qpos, get_qvel, etc.) are safe to call from any thread
/// - Write operations (step, set_qpos, etc.) must be externally synchronized
pub struct MuJoCoAdapter {
    /// Pointer to MuJoCo model (mjModel). Null when no model is loaded.
    model: *mut mjModel,
    /// Pointer to MuJoCo simulation data (mjData). Null when no model is loaded.
    data: *mut mjData,
    config: MuJoCoConfig,
}

// SAFETY: MuJoCo model and data can be safely transferred between threads.
// The MuJoCo library itself is thread-safe for read operations.
// Write operations (step, set_*, etc.) require external synchronization.
// We don't implement Sync because concurrent mutation is not safe.
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

        // SAFETY: All MuJoCo FFI calls here are safe because:
        // - c_path is a valid null-terminated C string
        // - error_buf is a properly sized stack buffer
        // - We check return values for null before using pointers
        // - On failure, we clean up any partially allocated resources
        unsafe {
            // Free existing model if any
            self.free_model();

            // Load new model from XML file
            // SAFETY: mj_loadXML reads from a file path and writes error to buffer.
            // Both pointers are valid and the buffer size is correct.
            let model = mj_loadXML(
                c_path.as_ptr(),
                ptr::null(),
                error_buf.as_mut_ptr(),
                error_buf.len() as i32,
            );

            if model.is_null() {
                // SAFETY: error_buf contains a null-terminated error message from MuJoCo
                let error_msg = CStr::from_ptr(error_buf.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                return Err(MuJoCoError::ModelLoadFailed(error_msg));
            }

            self.model = model;

            // Create simulation data structure for the loaded model
            // SAFETY: self.model is valid and non-null (checked above)
            let data = mj_makeData(self.model);
            if data.is_null() {
                // SAFETY: Cleaning up the model we just loaded
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
    ///
    /// Advances the physics simulation by one timestep using MuJoCo's mj_step.
    pub fn step(&mut self) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }

        // SAFETY: is_loaded() ensures both model and data are non-null and valid.
        // mj_step performs one simulation step and is safe to call with valid pointers.
        unsafe {
            mj_step(self.model, self.data);
        }

        Ok(())
    }

    /// Step forward kinematics only (no dynamics)
    ///
    /// Computes forward kinematics without integrating dynamics.
    /// Useful for visualization or computing derived quantities.
    pub fn forward(&mut self) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }

        // SAFETY: is_loaded() ensures valid pointers. mj_forward computes
        // forward kinematics and is safe with valid model/data.
        unsafe {
            mj_forward(self.model, self.data);
        }

        Ok(())
    }

    /// Reset simulation to initial state
    ///
    /// Resets all simulation state (positions, velocities, etc.) to initial values
    /// defined in the model.
    pub fn reset(&mut self) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }

        // SAFETY: is_loaded() ensures valid pointers. mj_resetData resets
        // the data structure and is safe with valid model/data.
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
    ///
    /// Returns a copy of the current joint positions as a Vec<f64>.
    /// The length equals model.nq (number of generalized coordinates).
    pub fn get_qpos(&self) -> Vec<f64> {
        if !self.is_loaded() {
            return Vec::new();
        }
        // SAFETY:
        // - is_loaded() ensures model and data are valid
        // - nq is the length of the qpos array as defined by MuJoCo
        // - We check qpos_ptr for null before creating a slice
        // - from_raw_parts requires nq elements starting at qpos_ptr, which
        //   MuJoCo guarantees for a properly loaded model
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
    ///
    /// Sets all joint positions. The input slice must have exactly model.nq elements.
    pub fn set_qpos(&mut self, qpos: &[f64]) -> Result<()> {
        if !self.is_loaded() {
            return Err(MuJoCoError::NoModel);
        }
        // SAFETY:
        // - is_loaded() ensures model and data are valid
        // - We validate that qpos.len() == nq before copying
        // - copy_nonoverlapping is safe because qpos and qpos_ptr don't overlap
        //   (qpos is on the Rust stack/heap, qpos_ptr is in MuJoCo's memory)
        // - Both regions have sufficient size (nq elements)
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
    ///
    /// Releases all MuJoCo resources. Safe to call multiple times.
    fn free_model(&mut self) {
        // SAFETY:
        // - We check each pointer for null before freeing
        // - After freeing, we set pointers to null to prevent double-free
        // - mj_deleteData and mj_deleteModel are safe to call with valid pointers
        // - Order matters: data depends on model, so free data first
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
