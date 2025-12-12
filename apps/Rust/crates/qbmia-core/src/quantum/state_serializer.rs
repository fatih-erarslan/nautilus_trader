//! Quantum state serialization and persistence

use crate::error::{QBMIAError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum state serializer for persistence
#[derive(Debug, Clone)]
pub struct QuantumStateSerializer {
    /// Stored quantum states
    states: HashMap<String, SerializedQuantumState>,
}

/// Serialized quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedQuantumState {
    /// State amplitudes
    pub amplitudes: Vec<f64>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Timestamp of creation
    pub timestamp: u64,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl QuantumStateSerializer {
    /// Create a new quantum state serializer
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }
    
    /// Serialize a quantum state
    pub fn serialize_state(
        &mut self,
        key: &str,
        amplitudes: &[f64],
        num_qubits: usize,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<()> {
        let state = SerializedQuantumState {
            amplitudes: amplitudes.to_vec(),
            num_qubits,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: metadata.unwrap_or_default(),
        };
        
        self.states.insert(key.to_string(), state);
        Ok(())
    }
    
    /// Deserialize a quantum state
    pub fn deserialize_state(&self, key: &str) -> Result<Array1<f64>> {
        let state = self.states.get(key)
            .ok_or_else(|| QBMIAError::quantum_simulation("State not found"))?;
        
        Ok(Array1::from_vec(state.amplitudes.clone()))
    }
    
    /// Serialize all states to JSON
    pub fn serialize_all_states(&self) -> Result<String> {
        serde_json::to_string(&self.states)
            .map_err(QBMIAError::from)
    }
    
    /// Restore all states from JSON
    pub fn restore_all_states(&mut self, json: &str) -> Result<()> {
        self.states = serde_json::from_str(json)
            .map_err(QBMIAError::from)?;
        Ok(())
    }
}

impl Default for QuantumStateSerializer {
    fn default() -> Self {
        Self::new()
    }
}