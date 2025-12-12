//! Core Traits and Interfaces for Quantum Computing
//!
//! This module defines the fundamental traits, types, and interfaces that enable
//! quantum-agentic-reasoning (QAR) integration with the quantum-core framework.

use crate::error::{QuantumError, QuantumResult};
use crate::quantum_state::QuantumState;
use crate::quantum_circuits::QuantumCircuit as CoreQuantumCircuit;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Circuit execution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitParams {
    /// Unique circuit identifier
    pub circuit_id: String,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// Number of measurement shots
    pub shots: usize,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Backend type to use for execution
    pub backend: String,
    /// Circuit depth limit
    pub max_depth: Option<usize>,
    /// Gate fidelity threshold
    pub fidelity_threshold: f64,
    /// Execution timeout in milliseconds
    pub timeout_ms: u64,
    /// Custom parameters for specific circuits
    pub custom_params: HashMap<String, f64>,
    /// Enable error correction
    pub error_correction: bool,
    /// Classical fallback enabled
    pub classical_fallback: bool,
}

impl Default for CircuitParams {
    fn default() -> Self {
        Self {
            circuit_id: Uuid::new_v4().to_string(),
            num_qubits: 4,
            shots: 1024,
            optimization_level: 2,
            backend: "simulator".to_string(),
            max_depth: None,
            fidelity_threshold: 0.99,
            timeout_ms: 5000,
            custom_params: HashMap::new(),
            error_correction: false,
            classical_fallback: true,
        }
    }
}

/// Execution context for quantum operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    /// Execution session ID
    pub session_id: String,
    /// Device configuration
    pub device_config: HashMap<String, String>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Performance metrics collection enabled
    pub collect_metrics: bool,
    /// Debug information level
    pub debug_level: u8,
    /// Execution environment
    pub environment: String,
    /// User preferences
    pub preferences: HashMap<String, String>,
    /// Execution timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            session_id: Uuid::new_v4().to_string(),
            device_config: HashMap::new(),
            resource_limits: ResourceLimits::default(),
            collect_metrics: true,
            debug_level: 1,
            environment: "development".to_string(),
            preferences: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Resource limits for quantum execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum CPU time in seconds
    pub max_cpu_time_s: u64,
    /// Maximum number of concurrent operations
    pub max_concurrent_ops: usize,
    /// Maximum circuit depth
    pub max_circuit_depth: usize,
    /// Maximum number of qubits
    pub max_qubits: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_time_s: 60,
            max_concurrent_ops: 4,
            max_circuit_depth: 1000,
            max_qubits: 32,
        }
    }
}

/// Quantum circuit execution trait
#[async_trait]
pub trait QuantumCircuit: Send + Sync {
    /// Execute the quantum circuit with given parameters
    async fn execute(
        &self,
        params: &CircuitParams,
        context: &ExecutionContext,
    ) -> QuantumResult<crate::QuantumResult>;

    /// Execute classical fallback if quantum execution fails
    async fn classical_fallback(&self, params: &CircuitParams) -> QuantumResult<crate::QuantumResult>;

    /// Validate circuit parameters before execution
    fn validate_parameters(&self, params: &CircuitParams) -> QuantumResult<()>;

    /// Get circuit complexity metrics
    fn complexity_metrics(&self) -> CircuitComplexity;

    /// Optimize circuit for target backend
    fn optimize(&mut self, backend: &str) -> QuantumResult<()>;

    /// Get circuit resource requirements
    fn resource_requirements(&self) -> ResourceRequirements;
}

/// Circuit complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitComplexity {
    /// Total number of gates
    pub gate_count: usize,
    /// Circuit depth
    pub depth: usize,
    /// Number of qubits used
    pub qubit_count: usize,
    /// Number of measurements
    pub measurement_count: usize,
    /// Estimated execution time in microseconds
    pub estimated_time_us: f64,
    /// Memory requirements in bytes
    pub memory_bytes: usize,
}

/// Resource requirements for circuit execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Required qubits
    pub qubits: usize,
    /// Required classical bits
    pub classical_bits: usize,
    /// Estimated memory usage in bytes
    pub memory_bytes: usize,
    /// Estimated execution time in microseconds
    pub execution_time_us: f64,
    /// Required backend capabilities
    pub backend_capabilities: Vec<String>,
}

/// Hardware interface trait for quantum devices
#[async_trait]
pub trait HardwareInterface: Send + Sync {
    /// Get device capabilities
    async fn capabilities(&self) -> QuantumResult<DeviceCapabilities>;

    /// Check device status
    async fn status(&self) -> QuantumResult<DeviceStatus>;

    /// Calibrate the device
    async fn calibrate(&mut self) -> QuantumResult<CalibrationResult>;

    /// Submit job to device
    async fn submit_job(&self, job: QuantumJob) -> QuantumResult<JobResult>;

    /// Cancel running job
    async fn cancel_job(&self, job_id: &str) -> QuantumResult<()>;

    /// Get job status
    async fn job_status(&self, job_id: &str) -> QuantumResult<JobStatus>;
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Maximum number of qubits
    pub max_qubits: usize,
    /// Supported gate types
    pub supported_gates: Vec<String>,
    /// Gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// Connectivity graph
    pub connectivity: Vec<(usize, usize)>,
    /// Coherence times in microseconds
    pub coherence_times: Vec<f64>,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
}

/// Device status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStatus {
    /// Is device online
    pub online: bool,
    /// Queue length
    pub queue_length: usize,
    /// Current utilization (0.0-1.0)
    pub utilization: f64,
    /// Error rate
    pub error_rate: f64,
    /// Last calibration time
    pub last_calibration: chrono::DateTime<chrono::Utc>,
    /// Status message
    pub message: String,
}

/// Calibration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    /// Calibration success
    pub success: bool,
    /// Updated gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// Coherence times
    pub coherence_times: Vec<f64>,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    /// Calibration timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Quantum job for hardware execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumJob {
    /// Job identifier
    pub id: String,
    /// Circuit to execute
    pub circuit: String, // Serialized circuit
    /// Job parameters
    pub params: CircuitParams,
    /// Priority level
    pub priority: JobPriority,
    /// Job metadata
    pub metadata: HashMap<String, String>,
}

/// Job priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Job execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    /// Job identifier
    pub job_id: String,
    /// Execution result
    pub result: crate::QuantumResult,
    /// Job status
    pub status: JobStatus,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Job execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Pattern recognition trait for quantum data analysis
#[async_trait]
pub trait PatternRecognizer: Send + Sync {
    /// Analyze quantum data for patterns
    async fn analyze(&self, data: &QuantumData) -> QuantumResult<PatternAnalysis>;

    /// Train pattern recognition model
    async fn train(&mut self, training_data: &[QuantumData]) -> QuantumResult<TrainingResult>;

    /// Predict patterns in new data
    async fn predict(&self, data: &QuantumData) -> QuantumResult<PatternPrediction>;

    /// Get model performance metrics
    fn performance_metrics(&self) -> ModelMetrics;
}

/// Quantum data for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumData {
    /// Data identifier
    pub id: String,
    /// Quantum states
    pub states: Vec<QuantumState>,
    /// Measurement results
    pub measurements: Vec<MeasurementResult>,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    /// Measured bits
    pub bits: Vec<u8>,
    /// Measurement probability
    pub probability: f64,
    /// Measurement count
    pub count: usize,
}

/// Pattern analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysis {
    /// Detected patterns
    pub patterns: Vec<DetectedPattern>,
    /// Confidence score
    pub confidence: f64,
    /// Analysis metadata
    pub metadata: HashMap<String, String>,
}

/// Detected pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern type
    pub pattern_type: String,
    /// Pattern strength (0.0-1.0)
    pub strength: f64,
    /// Pattern description
    pub description: String,
    /// Associated data
    pub data: HashMap<String, f64>,
}

/// Training result for pattern recognition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Training success
    pub success: bool,
    /// Final accuracy
    pub accuracy: f64,
    /// Training iterations
    pub iterations: usize,
    /// Training time in seconds
    pub training_time_s: f64,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

/// Pattern prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternPrediction {
    /// Predicted pattern
    pub pattern: DetectedPattern,
    /// Prediction confidence
    pub confidence: f64,
    /// Alternative predictions
    pub alternatives: Vec<DetectedPattern>,
}

/// Model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Accuracy score
    pub accuracy: f64,
    /// Precision score
    pub precision: f64,
    /// Recall score
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Training examples processed
    pub training_examples: usize,
    /// Model size in bytes
    pub model_size_bytes: usize,
}

/// Quantum algorithm execution trait
#[async_trait]
pub trait QuantumAlgorithm: Send + Sync {
    /// Execute the quantum algorithm
    async fn execute(&self, input: &AlgorithmInput) -> QuantumResult<AlgorithmOutput>;

    /// Get algorithm metadata
    fn metadata(&self) -> AlgorithmMetadata;

    /// Validate algorithm input
    fn validate_input(&self, input: &AlgorithmInput) -> QuantumResult<()>;

    /// Get resource requirements
    fn resource_requirements(&self, input: &AlgorithmInput) -> ResourceRequirements;
}

/// Algorithm input data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmInput {
    /// Input parameters
    pub parameters: HashMap<String, f64>,
    /// Input quantum states
    pub states: Vec<QuantumState>,
    /// Algorithm configuration
    pub config: HashMap<String, String>,
}

/// Algorithm output data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmOutput {
    /// Output quantum states
    pub states: Vec<QuantumState>,
    /// Classical results
    pub classical_results: Vec<f64>,
    /// Algorithm metrics
    pub metrics: HashMap<String, f64>,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Algorithm metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetadata {
    /// Algorithm name
    pub name: String,
    /// Algorithm version
    pub version: String,
    /// Algorithm description
    pub description: String,
    /// Author information
    pub author: String,
    /// Complexity class
    pub complexity: String,
    /// Quantum advantage
    pub quantum_advantage: bool,
}