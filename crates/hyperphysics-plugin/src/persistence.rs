//! Persistence utilities for saving/loading optimization results
//!
//! Requires the `serde` feature.

use std::fs;
use std::path::Path;

use crate::{HyperPhysicsError, Result, optimizer::OptimizationResult, swarm::SwarmResult};

/// Save an optimization result to a JSON file
pub fn save_result(result: &OptimizationResult, path: impl AsRef<Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))?;
    
    fs::write(path, json)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))?;
    
    Ok(())
}

/// Load an optimization result from a JSON file
pub fn load_result(path: impl AsRef<Path>) -> Result<OptimizationResult> {
    let json = fs::read_to_string(path)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))?;
    
    serde_json::from_str(&json)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))
}

/// Save a swarm result to a JSON file
pub fn save_swarm_result(result: &SwarmResult, path: impl AsRef<Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))?;
    
    fs::write(path, json)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))?;
    
    Ok(())
}

/// Load a swarm result from a JSON file
pub fn load_swarm_result(path: impl AsRef<Path>) -> Result<SwarmResult> {
    let json = fs::read_to_string(path)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))?;
    
    serde_json::from_str(&json)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))
}

/// Export results to CSV
pub fn export_convergence_csv(history: &[f64], path: impl AsRef<Path>) -> Result<()> {
    let mut csv = String::from("iteration,fitness\n");
    for (i, &fitness) in history.iter().enumerate() {
        csv.push_str(&format!("{},{}\n", i, fitness));
    }
    
    fs::write(path, csv)
        .map_err(|e| HyperPhysicsError::Serialization(e.to_string()))?;
    
    Ok(())
}
