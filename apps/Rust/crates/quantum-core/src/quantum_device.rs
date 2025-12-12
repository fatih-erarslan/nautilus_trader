//! Quantum Device Management System
//!
//! This module provides quantum device abstraction, management, and hardware
//! interface capabilities for quantum computing operations.

use crate::quantum_circuits::{QuantumCircuit, ExecutionResult};
use crate::error::{QuantumError, QuantumResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};
use chrono::{DateTime, Utc};

use uuid::Uuid;

/// Quantum device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Simulator,
    QuantumHardware,
    CloudQuantum,
    HybridClassical,
    NearTermDevice,
    FaultTolerantDevice,
}

/// Device backend implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceBackend {
    Qiskit,
    Cirq,
    PennyLane,
    Braket,
    IonQ,
    Rigetti,
    IBMQuantum,
    GoogleQuantum,
    CustomSimulator,
}

/// Device status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceStatus {
    Available,
    Busy,
    Maintenance,
    Error,
    Offline,
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Maximum number of qubits supported
    pub max_qubits: usize,
    /// Set of supported quantum gates
    pub gate_set: Vec<String>,
    /// Qubit connectivity graph (pairs of connected qubits)
    pub connectivity: Vec<(usize, usize)>,
    /// Error rates for different operations
    pub error_rates: HashMap<String, f64>,
    /// Coherence time in microseconds
    pub coherence_time_us: f64,
    /// Gate execution time in nanoseconds
    pub gate_time_ns: f64,
    /// Whether mid-circuit measurement is supported
    pub supports_mid_circuit_measurement: bool,
    /// Whether qubit reset is supported
    pub supports_reset: bool,
    /// Whether conditional operations are supported
    pub supports_conditional_operations: bool,
    /// Maximum number of shots per execution
    pub max_shots: usize,
    /// Whether parallel execution is supported
    pub parallel_execution: bool,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            max_qubits: 32,
            gate_set: vec![
                "H".to_string(),
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "CNOT".to_string(),
                "CZ".to_string(),
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "Phase".to_string(),
                "Toffoli".to_string(),
            ],
            connectivity: Vec::new(),
            error_rates: HashMap::new(),
            coherence_time_us: 100.0,
            gate_time_ns: 50.0,
            supports_mid_circuit_measurement: true,
            supports_reset: true,
            supports_conditional_operations: true,
            max_shots: 1000000,
            parallel_execution: true,
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Type of quantum device
    pub device_type: DeviceType,
    /// Backend implementation to use
    pub backend: DeviceBackend,
    /// Optional endpoint URL for remote devices
    pub endpoint: Option<String>,
    /// Optional authentication token
    pub authentication: Option<String>,
    /// Optional region for cloud devices
    pub region: Option<String>,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Number of retry attempts for failed requests
    pub retry_attempts: u32,
    /// Optional calibration data for the device
    pub calibration_data: Option<HashMap<String, f64>>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Simulator,
            backend: DeviceBackend::CustomSimulator,
            endpoint: None,
            authentication: None,
            region: None,
            timeout_ms: 30000,
            retry_attempts: 3,
            calibration_data: None,
        }
    }
}

/// Device metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetrics {
    /// Total number of executions
    pub total_executions: u64,
    /// Number of successful executions
    pub successful_executions: u64,
    /// Number of failed executions
    pub failed_executions: u64,
    /// Average execution time in milliseconds
    pub average_execution_time_ms: f64,
    /// Average quantum fidelity
    pub average_fidelity: f64,
    /// Current queue length
    pub queue_length: usize,
    /// Device utilization percentage
    pub utilization_percentage: f64,
    /// Timestamp of last calibration
    pub last_calibration: Option<DateTime<Utc>>,
    /// Device uptime in hours
    pub uptime_hours: f64,
}

impl Default for DeviceMetrics {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_execution_time_ms: 0.0,
            average_fidelity: 1.0,
            queue_length: 0,
            utilization_percentage: 0.0,
            last_calibration: None,
            uptime_hours: 0.0,
        }
    }
}

/// Main quantum device structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDevice {
    /// Unique device identifier
    pub id: String,
    /// Human-readable device name
    pub name: String,
    /// Type of quantum device
    pub device_type: DeviceType,
    /// Backend implementation used
    pub backend: DeviceBackend,
    /// Device capabilities and specifications
    pub capabilities: DeviceCapabilities,
    /// Device configuration settings
    pub config: DeviceConfig,
    /// Current device status
    pub status: DeviceStatus,
    /// Performance and usage metrics
    pub metrics: DeviceMetrics,
    /// Timestamp when device was created
    pub created_at: DateTime<Utc>,
    /// Timestamp of last update
    pub last_updated: DateTime<Utc>,
    /// Additional device metadata
    pub metadata: HashMap<String, String>,
}

impl QuantumDevice {
    /// Create a new quantum device
    pub fn new(
        name: String,
        device_type: DeviceType,
        backend: DeviceBackend,
        config: DeviceConfig,
    ) -> Self {
        let id = format!("device_{}", Uuid::new_v4());
        let now = Utc::now();
        
        Self {
            id,
            name,
            device_type,
            backend,
            capabilities: DeviceCapabilities::default(),
            config,
            status: DeviceStatus::Available,
            metrics: DeviceMetrics::default(),
            created_at: now,
            last_updated: now,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a simplified quantum device with just type and qubits
    pub fn new_simple(device_type: DeviceType, num_qubits: usize) -> QuantumResult<Self> {
        let mut capabilities = DeviceCapabilities::default();
        capabilities.max_qubits = num_qubits;
        
        let mut config = DeviceConfig::default();
        config.device_type = device_type;
        
        let device = Self::new(
            format!("{:?} Device", device_type),
            device_type,
            DeviceBackend::CustomSimulator,
            config,
        );
        
        Ok(device)
    }

    /// Update device status
    pub fn update_status(&mut self, status: DeviceStatus) {
        self.status = status;
        self.last_updated = Utc::now();
        
        info!("Device {} status updated to {:?}", self.id, status);
        // metrics::gauge!("quantum_device_status", status as u8 as f64);
    }

    /// Check if device is available
    pub fn is_available(&self) -> bool {
        matches!(self.status, DeviceStatus::Available)
    }

    /// Check if device supports a specific gate
    pub fn supports_gate(&self, gate: &str) -> bool {
        self.capabilities.gate_set.contains(&gate.to_string())
    }

    /// Check if device has sufficient qubits
    pub fn has_sufficient_qubits(&self, required: usize) -> bool {
        self.capabilities.max_qubits >= required
    }

    /// Get device error rate for a specific gate
    pub fn get_error_rate(&self, gate: &str) -> f64 {
        self.capabilities.error_rates.get(gate).copied().unwrap_or(0.001)
    }

    /// Update device metrics
    pub fn update_metrics(&mut self, execution_time_ms: f64, fidelity: f64, success: bool) {
        self.metrics.total_executions += 1;
        
        if success {
            self.metrics.successful_executions += 1;
        } else {
            self.metrics.failed_executions += 1;
        }
        
        // Update average execution time
        let total_time = self.metrics.average_execution_time_ms * (self.metrics.total_executions - 1) as f64;
        self.metrics.average_execution_time_ms = (total_time + execution_time_ms) / self.metrics.total_executions as f64;
        
        // Update average fidelity
        let total_fidelity = self.metrics.average_fidelity * (self.metrics.total_executions - 1) as f64;
        self.metrics.average_fidelity = (total_fidelity + fidelity) / self.metrics.total_executions as f64;
        
        self.last_updated = Utc::now();
        
        // Update metrics
        // metrics::counter!("quantum_device_executions_total", 1);
        // metrics::histogram!("quantum_device_execution_time_ms", execution_time_ms);
        // metrics::histogram!("quantum_device_fidelity", fidelity);
        
        debug!("Device {} metrics updated: {} executions, {:.3} avg fidelity", 
               self.id, self.metrics.total_executions, self.metrics.average_fidelity);
    }

    /// Calibrate device
    pub async fn calibrate(&mut self) -> QuantumResult<()> {
        info!("Starting calibration for device {}", self.id);
        
        self.update_status(DeviceStatus::Maintenance);
        
        // Simulate calibration process
        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
        
        // Update calibration data
        self.metrics.last_calibration = Some(Utc::now());
        
        // Reset error rates (simulated improvement)
        for (_gate, error_rate) in &mut self.capabilities.error_rates {
            *error_rate *= 0.9; // Improve error rates by 10%
        }
        
        self.update_status(DeviceStatus::Available);
        
        info!("Calibration completed for device {}", self.id);
        Ok(())
    }

    /// Get device health score
    pub fn get_health_score(&self) -> f64 {
        let success_rate = if self.metrics.total_executions > 0 {
            self.metrics.successful_executions as f64 / self.metrics.total_executions as f64
        } else {
            1.0
        };
        
        let fidelity_factor = self.metrics.average_fidelity;
        let uptime_factor = (self.metrics.uptime_hours / 24.0).min(1.0);
        
        (success_rate * 0.4 + fidelity_factor * 0.4 + uptime_factor * 0.2) * 100.0
    }

    /// Estimate execution time for a circuit
    pub fn estimate_execution_time(&self, circuit: &QuantumCircuit) -> f64 {
        let base_time = self.capabilities.gate_time_ns as f64 / 1_000_000.0; // Convert to ms
        let circuit_time = circuit.instructions.len() as f64 * base_time;
        let overhead = circuit_time * 0.1; // 10% overhead
        
        circuit_time + overhead
    }

    /// Check if device can execute circuit
    pub fn can_execute_circuit(&self, circuit: &QuantumCircuit) -> QuantumResult<bool> {
        // Check if device is available
        if !self.is_available() {
            return Ok(false);
        }
        
        // Check qubit requirements
        if !self.has_sufficient_qubits(circuit.num_qubits) {
            return Ok(false);
        }
        
        // Check gate support
        for instruction in &circuit.instructions {
            let gate_name = format!("{:?}", instruction.gate);
            if !self.supports_gate(&gate_name) {
                return Ok(false);
            }
        }
        
        // Check connectivity requirements (simplified)
        for instruction in &circuit.instructions {
            if instruction.qubits.len() > 1 {
                // Check if multi-qubit gates are supported
                let connectivity_ok = self.capabilities.connectivity.is_empty() || 
                    self.capabilities.connectivity.iter().any(|(c, t)| {
                        instruction.qubits.contains(c) && instruction.qubits.contains(t)
                    });
                
                if !connectivity_ok {
                    return Ok(false);
                }
            }
        }
        
        Ok(true)
    }
}

/// Device manager for managing multiple quantum devices
#[derive(Debug)]
pub struct DeviceManager {
    devices: Arc<RwLock<HashMap<String, QuantumDevice>>>,
    device_queue: Arc<Mutex<Vec<String>>>,
    #[allow(dead_code)]
    load_balancer: Arc<Mutex<LoadBalancer>>,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            device_queue: Arc::new(Mutex::new(Vec::new())),
            load_balancer: Arc::new(Mutex::new(LoadBalancer::new())),
        }
    }

    /// Register a new device
    pub async fn register_device(&self, device: QuantumDevice) -> QuantumResult<()> {
        let device_id = device.id.clone();
        
        {
            let mut devices = self.devices.write().await;
            devices.insert(device_id.clone(), device);
        }
        
        {
            let mut queue = self.device_queue.lock().unwrap();
            queue.push(device_id.clone());
        }
        
        info!("Device {} registered successfully", device_id);
        // metrics::counter!("quantum_devices_registered_total", 1);
        
        Ok(())
    }

    /// Unregister a device
    pub async fn unregister_device(&self, device_id: &str) -> QuantumResult<()> {
        {
            let mut devices = self.devices.write().await;
            devices.remove(device_id);
        }
        
        {
            let mut queue = self.device_queue.lock().unwrap();
            queue.retain(|id| id != device_id);
        }
        
        info!("Device {} unregistered", device_id);
        // metrics::counter!("quantum_devices_unregistered_total", 1);
        
        Ok(())
    }

    /// Get device by ID
    pub async fn get_device(&self, device_id: &str) -> Option<QuantumDevice> {
        let devices = self.devices.read().await;
        devices.get(device_id).cloned()
    }

    /// List all devices
    pub async fn list_devices(&self) -> Vec<QuantumDevice> {
        let devices = self.devices.read().await;
        devices.values().cloned().collect()
    }

    /// Find best device for circuit execution
    pub async fn find_best_device(&self, circuit: &QuantumCircuit) -> QuantumResult<Option<QuantumDevice>> {
        let devices = self.devices.read().await;
        let mut best_device = None;
        let mut best_score = 0.0;
        
        for device in devices.values() {
            if device.can_execute_circuit(circuit)? {
                let score = self.calculate_device_score(device, circuit);
                if score > best_score {
                    best_score = score;
                    best_device = Some(device.clone());
                }
            }
        }
        
        Ok(best_device)
    }

    /// Calculate device score for circuit execution
    fn calculate_device_score(&self, device: &QuantumDevice, circuit: &QuantumCircuit) -> f64 {
        let health_score = device.get_health_score() / 100.0;
        let execution_time = device.estimate_execution_time(circuit);
        let time_score = 1.0 / (1.0 + execution_time / 1000.0); // Normalize to 0-1
        let fidelity_score = device.metrics.average_fidelity;
        let availability_score = if device.is_available() { 1.0 } else { 0.0 };
        
        health_score * 0.3 + time_score * 0.2 + fidelity_score * 0.3 + availability_score * 0.2
    }

    /// Get device statistics
    pub async fn get_device_stats(&self) -> HashMap<String, DeviceMetrics> {
        let devices = self.devices.read().await;
        devices.iter()
            .map(|(id, device)| (id.clone(), device.metrics.clone()))
            .collect()
    }

    /// Update device status
    pub async fn update_device_status(&self, device_id: &str, status: DeviceStatus) -> QuantumResult<()> {
        let mut devices = self.devices.write().await;
        
        if let Some(device) = devices.get_mut(device_id) {
            device.update_status(status);
            info!("Device {} status updated to {:?}", device_id, status);
            Ok(())
        } else {
            Err(QuantumError::device_not_found(device_id.to_string()))
        }
    }

    /// Execute circuit on best available device
    pub async fn execute_circuit(&self, circuit: &mut QuantumCircuit) -> QuantumResult<ExecutionResult> {
        let device = self.find_best_device(circuit).await?
            .ok_or(QuantumError::no_available_device())?;
        
        info!("Executing circuit {} on device {}", circuit.id, device.id);
        
        // Update device status
        self.update_device_status(&device.id, DeviceStatus::Busy).await?;
        
        let start_time = std::time::Instant::now();
        
        // Execute circuit (simplified simulation)
        let execution_result = ExecutionResult {
            circuit_id: circuit.id.clone(),
            execution_time_ns: 0, // Will be updated
            memory_usage_bytes: circuit.estimate_memory_usage_public(),
            fidelity: device.metrics.average_fidelity * 0.95, // Slight degradation
            success: true,
            error_message: None,
            measurement_results: Some(vec![0; 1024]), // Dummy measurements
            timestamp: Utc::now(),
        };
        
        let execution_time = start_time.elapsed();
        
        // Update device metrics
        {
            let mut devices = self.devices.write().await;
            if let Some(device) = devices.get_mut(&device.id) {
                device.update_metrics(
                    execution_time.as_millis() as f64,
                    execution_result.fidelity,
                    execution_result.success,
                );
            }
        }
        
        // Update device status back to available
        self.update_device_status(&device.id, DeviceStatus::Available).await?;
        
        Ok(execution_result)
    }

    /// Calibrate all devices
    pub async fn calibrate_all_devices(&self) -> QuantumResult<()> {
        let device_ids: Vec<String> = {
            let devices = self.devices.read().await;
            devices.keys().cloned().collect()
        };
        
        for device_id in device_ids {
            let mut devices = self.devices.write().await;
            if let Some(device) = devices.get_mut(&device_id) {
                device.calibrate().await?;
            }
        }
        
        info!("All devices calibrated successfully");
        Ok(())
    }

    /// Get health report for all devices
    pub async fn get_health_report(&self) -> HashMap<String, f64> {
        let devices = self.devices.read().await;
        devices.iter()
            .map(|(id, device)| (id.clone(), device.get_health_score()))
            .collect()
    }
}

/// Load balancer for distributing circuits across devices
#[derive(Debug)]
struct LoadBalancer {
    #[allow(dead_code)]
    strategy: LoadBalancingStrategy,
    #[allow(dead_code)]
    device_loads: HashMap<String, f64>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    HealthBased,
    FidelityBased,
}

impl LoadBalancer {
    fn new() -> Self {
        Self {
            strategy: LoadBalancingStrategy::HealthBased,
            device_loads: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    fn select_device<'a>(&mut self, devices: &'a [QuantumDevice]) -> Option<&'a QuantumDevice> {
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                // Simple round-robin selection
                devices.first()
            }
            LoadBalancingStrategy::LeastLoaded => {
                // Select device with lowest load
                devices.iter().min_by(|a, b| {
                    let load_a = self.device_loads.get(&a.id).unwrap_or(&0.0);
                    let load_b = self.device_loads.get(&b.id).unwrap_or(&0.0);
                    load_a.partial_cmp(load_b).unwrap_or(std::cmp::Ordering::Equal)
                })
            }
            LoadBalancingStrategy::HealthBased => {
                // Select device with highest health score
                devices.iter().max_by(|a, b| {
                    a.get_health_score().partial_cmp(&b.get_health_score()).unwrap_or(std::cmp::Ordering::Equal)
                })
            }
            LoadBalancingStrategy::FidelityBased => {
                // Select device with highest fidelity
                devices.iter().max_by(|a, b| {
                    a.metrics.average_fidelity.partial_cmp(&b.metrics.average_fidelity).unwrap_or(std::cmp::Ordering::Equal)
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_circuits::CircuitBuilder;
    use tokio_test;

    #[test]
    fn test_device_creation() {
        let config = DeviceConfig::default();
        let device = QuantumDevice::new(
            "test_device".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            config,
        );
        
        assert_eq!(device.name, "test_device");
        assert_eq!(device.device_type, DeviceType::Simulator);
        assert_eq!(device.backend, DeviceBackend::CustomSimulator);
        assert!(device.is_available());
    }

    #[test]
    fn test_device_capabilities() {
        let device = QuantumDevice::new(
            "test".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        assert!(device.supports_gate("H"));
        assert!(device.supports_gate("CNOT"));
        assert!(!device.supports_gate("CUSTOM_GATE"));
        assert!(device.has_sufficient_qubits(10));
        assert!(!device.has_sufficient_qubits(100));
    }

    #[test]
    fn test_device_status_updates() {
        let mut device = QuantumDevice::new(
            "test".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        assert!(device.is_available());
        
        device.update_status(DeviceStatus::Busy);
        assert!(!device.is_available());
        assert_eq!(device.status, DeviceStatus::Busy);
        
        device.update_status(DeviceStatus::Available);
        assert!(device.is_available());
    }

    #[test]
    fn test_device_metrics() {
        let mut device = QuantumDevice::new(
            "test".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        assert_eq!(device.metrics.total_executions, 0);
        
        device.update_metrics(100.0, 0.95, true);
        assert_eq!(device.metrics.total_executions, 1);
        assert_eq!(device.metrics.successful_executions, 1);
        assert_eq!(device.metrics.average_execution_time_ms, 100.0);
        assert_eq!(device.metrics.average_fidelity, 0.95);
        
        device.update_metrics(200.0, 0.90, false);
        assert_eq!(device.metrics.total_executions, 2);
        assert_eq!(device.metrics.successful_executions, 1);
        assert_eq!(device.metrics.failed_executions, 1);
        assert_eq!(device.metrics.average_execution_time_ms, 150.0);
        assert_eq!(device.metrics.average_fidelity, 0.925);
    }

    #[test]
    fn test_device_health_score() {
        let mut device = QuantumDevice::new(
            "test".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        let initial_score = device.get_health_score();
        assert!(initial_score > 0.0);
        
        device.update_metrics(100.0, 0.95, true);
        device.update_metrics(200.0, 0.90, true);
        device.update_metrics(150.0, 0.92, false);
        
        let updated_score = device.get_health_score();
        assert!(updated_score > 0.0);
        assert!(updated_score < 100.0);
    }

    #[tokio::test]
    async fn test_device_calibration() {
        let mut device = QuantumDevice::new(
            "test".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        // Add some error rates
        device.capabilities.error_rates.insert("H".to_string(), 0.01);
        device.capabilities.error_rates.insert("CNOT".to_string(), 0.02);
        
        let initial_error = device.get_error_rate("H");
        assert!(device.calibrate().await.is_ok());
        
        let calibrated_error = device.get_error_rate("H");
        assert!(calibrated_error < initial_error);
        assert!(device.metrics.last_calibration.is_some());
    }

    #[test]
    fn test_execution_time_estimation() {
        let device = QuantumDevice::new(
            "test".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        let circuit = CircuitBuilder::new("test".to_string(), 2)
            .hadamard(0).unwrap()
            .cnot(0, 1).unwrap()
            .build();
        
        let estimated_time = device.estimate_execution_time(&circuit);
        assert!(estimated_time > 0.0);
    }

    #[test]
    fn test_circuit_execution_compatibility() {
        let device = QuantumDevice::new(
            "test".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        let compatible_circuit = CircuitBuilder::new("test".to_string(), 2)
            .hadamard(0).unwrap()
            .cnot(0, 1).unwrap()
            .build();
        
        assert!(device.can_execute_circuit(&compatible_circuit).unwrap());
        
        let incompatible_circuit = CircuitBuilder::new("test".to_string(), 100)
            .hadamard(0).unwrap()
            .build();
        
        assert!(!device.can_execute_circuit(&incompatible_circuit).unwrap());
    }

    #[tokio::test]
    async fn test_device_manager() {
        let manager = DeviceManager::new();
        
        let device1 = QuantumDevice::new(
            "device1".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        let device2 = QuantumDevice::new(
            "device2".to_string(),
            DeviceType::QuantumHardware,
            DeviceBackend::IBMQuantum,
            DeviceConfig::default(),
        );
        
        let device1_id = device1.id.clone();
        let device2_id = device2.id.clone();
        
        // Register devices
        assert!(manager.register_device(device1).await.is_ok());
        assert!(manager.register_device(device2).await.is_ok());
        
        // List devices
        let devices = manager.list_devices().await;
        assert_eq!(devices.len(), 2);
        
        // Get specific device
        let retrieved_device = manager.get_device(&device1_id).await;
        assert!(retrieved_device.is_some());
        assert_eq!(retrieved_device.unwrap().id, device1_id);
        
        // Unregister device
        assert!(manager.unregister_device(&device1_id).await.is_ok());
        
        let devices_after_unregister = manager.list_devices().await;
        assert_eq!(devices_after_unregister.len(), 1);
        assert_eq!(devices_after_unregister[0].id, device2_id);
    }

    #[tokio::test]
    async fn test_find_best_device() {
        let manager = DeviceManager::new();
        
        let mut device1 = QuantumDevice::new(
            "device1".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        device1.update_metrics(100.0, 0.95, true);
        
        let mut device2 = QuantumDevice::new(
            "device2".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        device2.update_metrics(200.0, 0.90, true);
        
        manager.register_device(device1).await.unwrap();
        manager.register_device(device2).await.unwrap();
        
        let circuit = CircuitBuilder::new("test".to_string(), 2)
            .hadamard(0).unwrap()
            .cnot(0, 1).unwrap()
            .build();
        
        let best_device = manager.find_best_device(&circuit).await.unwrap();
        assert!(best_device.is_some());
        
        let device = best_device.unwrap();
        assert_eq!(device.name, "device1"); // Should prefer higher fidelity
    }

    #[tokio::test]
    async fn test_execute_circuit_on_device() {
        let manager = DeviceManager::new();
        
        let device = QuantumDevice::new(
            "test_device".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        manager.register_device(device).await.unwrap();
        
        let mut circuit = CircuitBuilder::new("test".to_string(), 2)
            .hadamard(0).unwrap()
            .cnot(0, 1).unwrap()
            .build();
        
        let result = manager.execute_circuit(&mut circuit).await;
        assert!(result.is_ok());
        
        let execution_result = result.unwrap();
        assert!(execution_result.success);
        assert_eq!(execution_result.circuit_id, circuit.id);
        assert!(execution_result.measurement_results.is_some());
    }

    #[tokio::test]
    async fn test_device_stats() {
        let manager = DeviceManager::new();
        
        let device = QuantumDevice::new(
            "test_device".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        let device_id = device.id.clone();
        
        manager.register_device(device).await.unwrap();
        
        let mut circuit = CircuitBuilder::new("test".to_string(), 2)
            .hadamard(0).unwrap()
            .build();
        
        manager.execute_circuit(&mut circuit).await.unwrap();
        
        let stats = manager.get_device_stats().await;
        assert!(stats.contains_key(&device_id));
        assert_eq!(stats[&device_id].total_executions, 1);
        assert_eq!(stats[&device_id].successful_executions, 1);
    }

    #[tokio::test]
    async fn test_health_report() {
        let manager = DeviceManager::new();
        
        let device = QuantumDevice::new(
            "test_device".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        let device_id = device.id.clone();
        
        manager.register_device(device).await.unwrap();
        
        let health_report = manager.get_health_report().await;
        assert!(health_report.contains_key(&device_id));
        assert!(health_report[&device_id] > 0.0);
    }

    #[tokio::test]
    async fn test_calibrate_all_devices() {
        let manager = DeviceManager::new();
        
        let device1 = QuantumDevice::new(
            "device1".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        let device2 = QuantumDevice::new(
            "device2".to_string(),
            DeviceType::Simulator,
            DeviceBackend::CustomSimulator,
            DeviceConfig::default(),
        );
        
        manager.register_device(device1).await.unwrap();
        manager.register_device(device2).await.unwrap();
        
        assert!(manager.calibrate_all_devices().await.is_ok());
    }

    #[test]
    fn test_device_types_and_backends() {
        let types = vec![
            DeviceType::Simulator,
            DeviceType::QuantumHardware,
            DeviceType::CloudQuantum,
            DeviceType::HybridClassical,
            DeviceType::NearTermDevice,
            DeviceType::FaultTolerantDevice,
        ];
        
        let backends = vec![
            DeviceBackend::Qiskit,
            DeviceBackend::Cirq,
            DeviceBackend::PennyLane,
            DeviceBackend::Braket,
            DeviceBackend::IonQ,
            DeviceBackend::Rigetti,
            DeviceBackend::IBMQuantum,
            DeviceBackend::GoogleQuantum,
            DeviceBackend::CustomSimulator,
        ];
        
        // Test serialization/deserialization
        for device_type in types {
            let serialized = serde_json::to_string(&device_type).unwrap();
            let deserialized: DeviceType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(device_type, deserialized);
        }
        
        for backend in backends {
            let serialized = serde_json::to_string(&backend).unwrap();
            let deserialized: DeviceBackend = serde_json::from_str(&serialized).unwrap();
            assert_eq!(backend, deserialized);
        }
    }

    #[test]
    fn test_device_config_defaults() {
        let config = DeviceConfig::default();
        assert_eq!(config.device_type, DeviceType::Simulator);
        assert_eq!(config.backend, DeviceBackend::CustomSimulator);
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.retry_attempts, 3);
    }

    #[test]
    fn test_device_capabilities_defaults() {
        let capabilities = DeviceCapabilities::default();
        assert_eq!(capabilities.max_qubits, 32);
        assert!(capabilities.gate_set.contains(&"H".to_string()));
        assert!(capabilities.gate_set.contains(&"CNOT".to_string()));
        assert_eq!(capabilities.coherence_time_us, 100.0);
        assert_eq!(capabilities.gate_time_ns, 50.0);
        assert!(capabilities.supports_mid_circuit_measurement);
    }
}