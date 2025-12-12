//! Quantum backend implementations and hardware abstraction
//!
//! This module provides quantum backend implementations including:
//! - Quantum simulator backend
//! - Hardware abstraction layer
//! - Backend management and fallback strategies

use crate::core::{QarResult, QarError, constants, CircuitParams, ExecutionContext, QuantumResult, HardwareInterface, HardwareCapabilities, HardwareMetrics};
use crate::quantum::{QuantumState, Gate};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use super::types::*;
use super::traits::*;
use crate::core::traits::QuantumCircuit;

/// Quantum backend types
#[derive(Debug, Clone, PartialEq)]
pub enum BackendType {
    /// Local quantum simulator
    Simulator,
    /// IBM Quantum backend
    IbmQuantum,
    /// Google Quantum AI
    GoogleQuantum,
    /// Amazon Braket
    AmazonBraket,
    /// Rigetti Computing
    Rigetti,
    /// Microsoft Azure Quantum
    AzureQuantum,
    /// Classical fallback
    Classical,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Simulator => write!(f, "simulator"),
            BackendType::IbmQuantum => write!(f, "ibm_quantum"),
            BackendType::GoogleQuantum => write!(f, "google_quantum"),
            BackendType::AmazonBraket => write!(f, "amazon_braket"),
            BackendType::Rigetti => write!(f, "rigetti"),
            BackendType::AzureQuantum => write!(f, "azure_quantum"),
            BackendType::Classical => write!(f, "classical"),
        }
    }
}

/// Quantum backend configuration
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Backend type
    pub backend_type: BackendType,
    /// Maximum qubits available
    pub max_qubits: usize,
    /// Backend-specific parameters
    pub parameters: HashMap<String, String>,
    /// Timeout for operations
    pub timeout_ms: u64,
    /// Whether the backend is available
    pub available: bool,
}

impl BackendConfig {
    /// Create a new backend configuration
    pub fn new(backend_type: BackendType, max_qubits: usize) -> Self {
        Self {
            backend_type,
            max_qubits,
            parameters: HashMap::new(),
            timeout_ms: constants::QUANTUM_EXECUTION_TIMEOUT_MS,
            available: true,
        }
    }

    /// Add a parameter
    pub fn with_parameter(mut self, key: String, value: String) -> Self {
        self.parameters.insert(key, value);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set availability
    pub fn with_availability(mut self, available: bool) -> Self {
        self.available = available;
        self
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::new(BackendType::Simulator, constants::DEFAULT_NUM_QUBITS)
    }
}

/// Quantum simulator backend
#[derive(Debug)]
pub struct SimulatorBackend {
    /// Backend configuration
    config: BackendConfig,
    /// Execution statistics
    stats: Arc<RwLock<ExecutionStats>>,
}

impl SimulatorBackend {
    /// Create a new simulator backend
    pub fn new(config: BackendConfig) -> Self {
        Self {
            config,
            stats: Arc::new(RwLock::new(ExecutionStats::new())),
        }
    }

    /// Execute a quantum circuit on the simulator
    pub async fn execute_circuit(
        &self,
        circuit: &dyn QuantumCircuit,
        params: &CircuitParams,
        context: &ExecutionContext,
    ) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();
        
        // Check if we can handle this circuit
        if circuit.num_qubits() > self.config.max_qubits {
            return Err(QarError::QuantumError(
                format!("Circuit requires {} qubits, but backend only supports {}", 
                       circuit.num_qubits(), self.config.max_qubits)
            ));
        }

        // Validate parameters
        circuit.validate_parameters(params)?;

        // Execute the circuit
        let result = match tokio::time::timeout(
            std::time::Duration::from_millis(context.max_execution_time_ms),
            circuit.execute(params, context)
        ).await {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(QarError::QuantumError("Circuit execution timeout".to_string())),
        };

        // Update statistics
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let mut stats = self.stats.write().await;
        stats.record_execution(execution_time, circuit.num_qubits(), true);

        Ok(result)
    }

    /// Get backend statistics
    pub async fn get_stats(&self) -> ExecutionStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = ExecutionStats::new();
    }
}

#[async_trait]
impl HardwareInterface for SimulatorBackend {
    async fn is_quantum_available(&self) -> bool {
        self.config.available && self.config.backend_type != BackendType::Classical
    }

    async fn get_quantum_backends(&self) -> Vec<String> {
        if self.config.available {
            vec![self.config.backend_type.to_string()]
        } else {
            vec![]
        }
    }

    async fn execute_quantum(
        &self,
        circuit: &dyn QuantumCircuit,
        params: &CircuitParams,
    ) -> QarResult<QuantumResult> {
        let context = ExecutionContext::default();
        self.execute_circuit(circuit, params, &context).await
    }

    async fn get_capabilities(&self) -> HardwareCapabilities {
        HardwareCapabilities {
            max_qubits: self.config.max_qubits,
            quantum_backends: vec![self.config.backend_type.to_string()],
            gpu_available: cfg!(feature = "gpu"),
            supported_gates: vec![
                "X".to_string(), "Y".to_string(), "Z".to_string(),
                "H".to_string(), "S".to_string(), "T".to_string(),
                "RX".to_string(), "RY".to_string(), "RZ".to_string(),
                "CNOT".to_string(), "CZ".to_string(), "SWAP".to_string(),
            ],
        }
    }

    async fn get_metrics(&self) -> HardwareMetrics {
        let stats = self.stats.read().await;
        
        HardwareMetrics {
            quantum_time_ms: stats.total_quantum_time_ms,
            classical_time_ms: stats.total_classical_time_ms,
            memory_usage_mb: stats.peak_memory_usage_mb,
            gpu_utilization: 0.0, // Not implemented for simulator
            quantum_gates: stats.total_quantum_gates,
            cache_hit_ratio: stats.cache_hit_ratio,
        }
    }
}

/// Hardware backend for real quantum devices
#[derive(Debug)]
pub struct HardwareBackend {
    /// Backend configuration
    config: BackendConfig,
    /// Fallback simulator
    fallback: Option<SimulatorBackend>,
    /// Connection status
    connected: Arc<RwLock<bool>>,
    /// Execution statistics
    stats: Arc<RwLock<ExecutionStats>>,
}

impl HardwareBackend {
    /// Create a new hardware backend
    pub fn new(config: BackendConfig) -> Self {
        let fallback = Some(SimulatorBackend::new(
            BackendConfig::new(BackendType::Simulator, config.max_qubits)
        ));

        Self {
            config,
            fallback,
            connected: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(ExecutionStats::new())),
        }
    }

    /// Connect to the quantum hardware
    pub async fn connect(&self) -> QarResult<()> {
        // Simulate connection process
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        // In a real implementation, this would establish connection to quantum hardware
        let success = match self.config.backend_type {
            BackendType::IbmQuantum => self.connect_ibm().await?,
            BackendType::GoogleQuantum => self.connect_google().await?,
            BackendType::AmazonBraket => self.connect_braket().await?,
            BackendType::Rigetti => self.connect_rigetti().await?,
            BackendType::AzureQuantum => self.connect_azure().await?,
            _ => false,
        };

        let mut connected = self.connected.write().await;
        *connected = success;

        if success {
            Ok(())
        } else {
            Err(QarError::HardwareError(
                format!("Failed to connect to {} backend", self.config.backend_type)
            ))
        }
    }

    /// Disconnect from quantum hardware
    pub async fn disconnect(&self) {
        let mut connected = self.connected.write().await;
        *connected = false;
    }

    /// Execute circuit on hardware with fallback
    pub async fn execute_with_fallback(
        &self,
        circuit: &dyn QuantumCircuit,
        params: &CircuitParams,
        context: &ExecutionContext,
    ) -> QarResult<QuantumResult> {
        let connected = *self.connected.read().await;
        
        if connected && context.prefer_quantum {
            // Try quantum hardware first
            match self.execute_on_hardware(circuit, params, context).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    log::warn!("Quantum hardware execution failed: {}, falling back to simulator", e);
                }
            }
        }

        // Use fallback simulator
        if let Some(ref fallback) = self.fallback {
            if circuit.supports_classical_fallback() {
                // Try classical fallback first
                match circuit.classical_fallback(params).await {
                    Ok(result) => Ok(result),
                    Err(_) => fallback.execute_circuit(circuit, params, context).await,
                }
            } else {
                fallback.execute_circuit(circuit, params, context).await
            }
        } else {
            Err(QarError::HardwareError("No fallback available".to_string()))
        }
    }

    /// Execute circuit directly on quantum hardware
    async fn execute_on_hardware(
        &self,
        circuit: &dyn QuantumCircuit,
        params: &CircuitParams,
        _context: &ExecutionContext,
    ) -> QarResult<QuantumResult> {
        let start_time = std::time::Instant::now();
        
        // Simulate hardware execution
        // In real implementation, this would submit job to quantum device
        tokio::time::sleep(std::time::Duration::from_millis(
            circuit.estimated_execution_time_ms()
        )).await;

        // For now, return a simulated result
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let expectation_values = vec![0.5; circuit.num_qubits()]; // Mock result

        let mut stats = self.stats.write().await;
        stats.record_execution(execution_time, circuit.num_qubits(), true);

        Ok(QuantumResult::new(expectation_values, execution_time, true)
            .with_metadata("backend".to_string(), self.config.backend_type.to_string())
            .with_metadata("hardware".to_string(), "true".to_string()))
    }

    // Hardware-specific connection methods
    async fn connect_ibm(&self) -> QarResult<bool> {
        // Simulate IBM Quantum connection
        log::info!("Connecting to IBM Quantum...");
        Ok(false) // Not actually connected in this demo
    }

    async fn connect_google(&self) -> QarResult<bool> {
        // Simulate Google Quantum AI connection
        log::info!("Connecting to Google Quantum AI...");
        Ok(false) // Not actually connected in this demo
    }

    async fn connect_braket(&self) -> QarResult<bool> {
        // Simulate Amazon Braket connection
        log::info!("Connecting to Amazon Braket...");
        Ok(false) // Not actually connected in this demo
    }

    async fn connect_rigetti(&self) -> QarResult<bool> {
        // Simulate Rigetti connection
        log::info!("Connecting to Rigetti Computing...");
        Ok(false) // Not actually connected in this demo
    }

    async fn connect_azure(&self) -> QarResult<bool> {
        // Simulate Azure Quantum connection
        log::info!("Connecting to Azure Quantum...");
        Ok(false) // Not actually connected in this demo
    }
}

#[async_trait]
impl HardwareInterface for HardwareBackend {
    async fn is_quantum_available(&self) -> bool {
        *self.connected.read().await && self.config.available
    }

    async fn get_quantum_backends(&self) -> Vec<String> {
        if self.config.available {
            vec![self.config.backend_type.to_string()]
        } else {
            vec![]
        }
    }

    async fn execute_quantum(
        &self,
        circuit: &dyn QuantumCircuit,
        params: &CircuitParams,
    ) -> QarResult<QuantumResult> {
        let context = ExecutionContext::default();
        self.execute_with_fallback(circuit, params, &context).await
    }

    async fn get_capabilities(&self) -> HardwareCapabilities {
        let mut capabilities = HardwareCapabilities {
            max_qubits: self.config.max_qubits,
            quantum_backends: vec![self.config.backend_type.to_string()],
            gpu_available: cfg!(feature = "gpu"),
            supported_gates: Vec::new(),
        };

        // Add backend-specific gate support
        capabilities.supported_gates = match self.config.backend_type {
            BackendType::IbmQuantum => vec![
                "X".to_string(), "Y".to_string(), "Z".to_string(),
                "H".to_string(), "S".to_string(), "T".to_string(),
                "RX".to_string(), "RY".to_string(), "RZ".to_string(),
                "CNOT".to_string(), "CZ".to_string(),
            ],
            BackendType::GoogleQuantum => vec![
                "X".to_string(), "Y".to_string(), "Z".to_string(),
                "H".to_string(), "RX".to_string(), "RY".to_string(),
                "CZ".to_string(), "ISWAP".to_string(),
            ],
            BackendType::Rigetti => vec![
                "X".to_string(), "Y".to_string(), "Z".to_string(),
                "RX".to_string(), "RY".to_string(), "RZ".to_string(),
                "CZ".to_string(), "CCNOT".to_string(),
            ],
            _ => vec![
                "X".to_string(), "Y".to_string(), "Z".to_string(),
                "H".to_string(), "CNOT".to_string(), "CZ".to_string(),
            ],
        };

        capabilities
    }

    async fn get_metrics(&self) -> HardwareMetrics {
        let stats = self.stats.read().await;
        
        HardwareMetrics {
            quantum_time_ms: stats.total_quantum_time_ms,
            classical_time_ms: stats.total_classical_time_ms,
            memory_usage_mb: stats.peak_memory_usage_mb,
            gpu_utilization: 0.0, // Would be measured from actual hardware
            quantum_gates: stats.total_quantum_gates,
            cache_hit_ratio: stats.cache_hit_ratio,
        }
    }
}

/// Backend manager for handling multiple quantum backends
#[derive(Debug)]
pub struct BackendManager {
    /// Available backends
    backends: HashMap<String, Box<dyn HardwareInterface + Send + Sync>>,
    /// Default backend name
    default_backend: String,
    /// Backend selection strategy
    selection_strategy: BackendSelectionStrategy,
}

impl BackendManager {
    /// Create a new backend manager
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            default_backend: "simulator".to_string(),
            selection_strategy: BackendSelectionStrategy::PreferQuantum,
        }
    }

    /// Add a backend
    pub fn add_backend(
        &mut self,
        name: String,
        backend: Box<dyn HardwareInterface + Send + Sync>,
    ) {
        self.backends.insert(name, backend);
    }

    /// Set default backend
    pub fn set_default_backend(&mut self, name: String) {
        self.default_backend = name;
    }

    /// Set backend selection strategy
    pub fn set_selection_strategy(&mut self, strategy: BackendSelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// Select best backend for a circuit
    pub async fn select_backend(&self, circuit: &dyn QuantumCircuit) -> Option<&str> {
        match self.selection_strategy {
            BackendSelectionStrategy::PreferQuantum => {
                // Try to find available quantum backend
                for (name, backend) in &self.backends {
                    if backend.is_quantum_available().await {
                        let capabilities = backend.get_capabilities().await;
                        if capabilities.max_qubits >= circuit.num_qubits() {
                            return Some(name);
                        }
                    }
                }
                // Fall back to default
                Some(&self.default_backend)
            }
            BackendSelectionStrategy::PreferSimulator => {
                Some(&self.default_backend)
            }
            BackendSelectionStrategy::OptimalPerformance => {
                // Select based on estimated execution time and capabilities
                let mut best_backend = None;
                let mut best_score = f64::INFINITY;

                for (name, backend) in &self.backends {
                    let capabilities = backend.get_capabilities().await;
                    if capabilities.max_qubits >= circuit.num_qubits() {
                        let metrics = backend.get_metrics().await;
                        let score = circuit.estimated_execution_time_ms() as f64
                                  + metrics.quantum_time_ms
                                  - (metrics.cache_hit_ratio * 1000.0); // Prefer cached results

                        if score < best_score {
                            best_score = score;
                            best_backend = Some(name.as_str());
                        }
                    }
                }

                best_backend.or(Some(&self.default_backend))
            }
        }
    }

    /// Execute circuit on selected backend
    pub async fn execute_circuit(
        &self,
        circuit: &dyn QuantumCircuit,
        params: &CircuitParams,
    ) -> QarResult<QuantumResult> {
        let backend_name = self.select_backend(circuit).await
            .ok_or_else(|| QarError::HardwareError("No suitable backend available".to_string()))?;

        let backend = self.backends.get(backend_name)
            .ok_or_else(|| QarError::HardwareError(format!("Backend {} not found", backend_name)))?;

        backend.execute_quantum(circuit, params).await
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend selection strategies
#[derive(Debug, Clone, PartialEq)]
pub enum BackendSelectionStrategy {
    /// Prefer quantum hardware when available
    PreferQuantum,
    /// Always use simulator
    PreferSimulator,
    /// Select based on optimal performance
    OptimalPerformance,
}

/// Execution statistics
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Total quantum execution time
    pub total_quantum_time_ms: f64,
    /// Total classical execution time
    pub total_classical_time_ms: f64,
    /// Peak memory usage
    pub peak_memory_usage_mb: f64,
    /// Total quantum gates executed
    pub total_quantum_gates: usize,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Number of executions
    pub execution_count: usize,
}

impl ExecutionStats {
    /// Create new execution statistics
    pub fn new() -> Self {
        Self {
            total_quantum_time_ms: 0.0,
            total_classical_time_ms: 0.0,
            peak_memory_usage_mb: 0.0,
            total_quantum_gates: 0,
            cache_hit_ratio: 0.0,
            execution_count: 0,
        }
    }

    /// Record an execution
    pub fn record_execution(&mut self, time_ms: f64, gates: usize, is_quantum: bool) {
        if is_quantum {
            self.total_quantum_time_ms += time_ms;
        } else {
            self.total_classical_time_ms += time_ms;
        }
        self.total_quantum_gates += gates;
        self.execution_count += 1;
    }

    /// Get average execution time
    pub fn average_execution_time_ms(&self) -> f64 {
        if self.execution_count > 0 {
            (self.total_quantum_time_ms + self.total_classical_time_ms) / self.execution_count as f64
        } else {
            0.0
        }
    }
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config() {
        let config = BackendConfig::new(BackendType::Simulator, 8)
            .with_parameter("shots".to_string(), "1000".to_string())
            .with_timeout(5000)
            .with_availability(true);

        assert_eq!(config.backend_type, BackendType::Simulator);
        assert_eq!(config.max_qubits, 8);
        assert_eq!(config.timeout_ms, 5000);
        assert!(config.available);
        assert_eq!(config.parameters.get("shots"), Some(&"1000".to_string()));
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::Simulator.to_string(), "simulator");
        assert_eq!(BackendType::IbmQuantum.to_string(), "ibm_quantum");
        assert_eq!(BackendType::Classical.to_string(), "classical");
    }

    #[tokio::test]
    async fn test_simulator_backend() {
        let config = BackendConfig::new(BackendType::Simulator, 4);
        let backend = SimulatorBackend::new(config);

        assert!(backend.is_quantum_available().await);
        
        let backends = backend.get_quantum_backends().await;
        assert_eq!(backends, vec!["simulator"]);
        
        let capabilities = backend.get_capabilities().await;
        assert_eq!(capabilities.max_qubits, 4);
        assert!(!capabilities.supported_gates.is_empty());
    }

    #[tokio::test]
    async fn test_hardware_backend() {
        let config = BackendConfig::new(BackendType::IbmQuantum, 8);
        let backend = HardwareBackend::new(config);

        // Should not be connected initially
        assert!(!backend.is_quantum_available().await);
        
        let capabilities = backend.get_capabilities().await;
        assert_eq!(capabilities.max_qubits, 8);
    }

    #[test]
    fn test_backend_manager() {
        let mut manager = BackendManager::new();
        assert_eq!(manager.default_backend, "simulator");
        
        manager.set_default_backend("test_backend".to_string());
        assert_eq!(manager.default_backend, "test_backend");
        
        manager.set_selection_strategy(BackendSelectionStrategy::PreferSimulator);
        assert_eq!(manager.selection_strategy, BackendSelectionStrategy::PreferSimulator);
    }

    #[test]
    fn test_execution_stats() {
        let mut stats = ExecutionStats::new();
        
        stats.record_execution(100.0, 10, true);
        stats.record_execution(50.0, 5, false);
        
        assert_eq!(stats.total_quantum_time_ms, 100.0);
        assert_eq!(stats.total_classical_time_ms, 50.0);
        assert_eq!(stats.total_quantum_gates, 15);
        assert_eq!(stats.execution_count, 2);
        assert_eq!(stats.average_execution_time_ms(), 75.0);
    }
}