//! Quantum Hardware Module
//!
//! PennyLane-compatible quantum hardware abstraction for trading operations with device management.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use crate::quantum::{QuantumCircuit, QuantumState, gates::Gate};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Lightning device hierarchy (PennyLane compatible)
/// Following user preference: lightning.gpu -> lightning.kokkos -> lightning.qubit
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceType {
    // Lightning hierarchy (performance ordered)
    LightningGpu,    // Fastest: GPU-accelerated
    LightningKokkos, // Medium: CPU-optimized with Kokkos
    LightningQubit,  // Basic: Standard CPU implementation
    
    // Legacy Lightning (maps to LightningQubit)
    Lightning,
    
    // Custom devices
    Custom(String),
}

/// Quantum device status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceStatus {
    Online,
    Offline,
    Maintenance,
    Busy,
    Error,
    Calibrating,
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub max_qubits: usize,
    pub max_shots: u64,
    pub gate_set: Vec<String>,
    pub connectivity: Vec<(usize, usize)>, // Qubit connectivity graph
    pub noise_model: Option<NoiseModel>,
    pub coherence_time_us: Option<f64>,
    pub gate_fidelity: Option<f64>,
    pub readout_fidelity: Option<f64>,
    pub supports_measurements: bool,
    pub supports_conditional: bool,
    pub supports_reset: bool,
}

/// Noise model for realistic simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseModel {
    pub depolarizing_error: Option<f64>,
    pub thermal_relaxation: Option<ThermalRelaxation>,
    pub gate_errors: HashMap<String, f64>,
    pub readout_errors: Vec<Vec<f64>>, // Confusion matrix
    pub crosstalk_errors: HashMap<String, f64>,
}

/// Thermal relaxation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalRelaxation {
    pub t1_time_us: f64, // Relaxation time
    pub t2_time_us: f64, // Dephasing time
    pub excited_state_population: f64,
}

/// Quantum device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_type: DeviceType,
    pub backend_name: Option<String>,
    pub shots: u64,
    pub seed: Option<u64>,
    pub optimization_level: u8,
    pub initial_layout: Option<Vec<usize>>,
    pub coupling_map: Option<Vec<(usize, usize)>>,
    pub basis_gates: Option<Vec<String>>,
    pub noise_model: Option<NoiseModel>,
    pub memory: bool,
    pub max_parallel_experiments: u32,
    pub provider_config: HashMap<String, String>,
}

/// Quantum job for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumJob {
    pub id: String,
    pub circuit: QuantumCircuit,
    pub device_type: DeviceType,
    pub shots: u64,
    pub metadata: HashMap<String, String>,
    pub submitted_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: JobStatus,
    pub result: Option<JobResult>,
    pub error_message: Option<String>,
    pub queue_position: Option<u32>,
}

/// Job execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Job execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub counts: HashMap<String, u64>,
    pub probabilities: Option<Vec<f64>>,
    pub statevector: Option<Vec<num_complex::Complex64>>,
    pub expectation_values: Option<Vec<f64>>,
    pub execution_time_ms: u64,
    pub shots_used: u64,
    pub fidelity: Option<f64>,
    pub metadata: HashMap<String, String>,
}

/// Quantum device abstraction
#[derive(Debug)]
pub struct QuantumDevice {
    pub device_type: DeviceType,
    pub name: String,
    pub config: DeviceConfig,
    pub capabilities: DeviceCapabilities,
    pub status: Arc<RwLock<DeviceStatus>>,
    pub job_queue: Arc<RwLock<Vec<QuantumJob>>>,
    pub active_jobs: Arc<RwLock<HashMap<String, QuantumJob>>>,
    pub completed_jobs: Arc<RwLock<Vec<QuantumJob>>>,
    pub device_executor: Arc<dyn DeviceExecutor + Send + Sync>,
    pub calibration_data: Arc<RwLock<CalibrationData>>,
    pub usage_stats: Arc<Mutex<DeviceUsageStats>>,
}

/// Device executor trait for different backends
#[async_trait::async_trait]
pub trait DeviceExecutor {
    async fn execute_circuit(&self, circuit: &QuantumCircuit, shots: u64, config: &DeviceConfig) -> QarResult<JobResult>;
    async fn get_device_status(&self) -> QarResult<DeviceStatus>;
    async fn calibrate_device(&self) -> QarResult<CalibrationData>;
    async fn estimate_execution_time(&self, circuit: &QuantumCircuit, shots: u64) -> QarResult<u64>;
    fn get_capabilities(&self) -> DeviceCapabilities;
}

/// Device calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    pub calibrated_at: DateTime<Utc>,
    pub gate_fidelities: HashMap<String, f64>,
    pub qubit_frequencies: Vec<f64>,
    pub coupling_strengths: HashMap<(usize, usize), f64>,
    pub readout_fidelities: Vec<f64>,
    pub coherence_times: Vec<(f64, f64)>, // (T1, T2) for each qubit
    pub cross_talk_matrix: Vec<Vec<f64>>,
    pub temperature_mk: Option<f64>,
}

/// Device usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceUsageStats {
    pub total_jobs: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub total_shots: u64,
    pub total_execution_time_ms: u64,
    pub average_queue_time_ms: f64,
    pub average_execution_time_ms: f64,
    pub utilization_percentage: f64,
    pub last_updated: DateTime<Utc>,
}

/// Quantum hardware manager
#[derive(Debug)]
pub struct QuantumHardwareManager {
    devices: Arc<RwLock<HashMap<String, QuantumDevice>>>,
    device_registry: Arc<RwLock<HashMap<DeviceType, Vec<String>>>>,
    job_scheduler: Arc<dyn JobScheduler + Send + Sync>,
    monitoring_service: Arc<dyn DeviceMonitoringService + Send + Sync>,
    calibration_scheduler: Arc<dyn CalibrationScheduler + Send + Sync>,
}

/// Job scheduling trait
#[async_trait::async_trait]
pub trait JobScheduler {
    async fn schedule_job(&self, job: QuantumJob, available_devices: &[String]) -> QarResult<String>;
    async fn get_optimal_device(&self, circuit: &QuantumCircuit, requirements: &JobRequirements) -> QarResult<String>;
    async fn estimate_queue_time(&self, device_id: &str) -> QarResult<u64>;
}

/// Device monitoring service trait
#[async_trait::async_trait]
pub trait DeviceMonitoringService {
    async fn monitor_device_health(&self, device_id: &str) -> QarResult<DeviceHealthReport>;
    async fn collect_device_metrics(&self, device_id: &str) -> QarResult<DeviceMetrics>;
    async fn detect_anomalies(&self, device_id: &str) -> QarResult<Vec<DeviceAnomaly>>;
}

/// Calibration scheduling trait
#[async_trait::async_trait]
pub trait CalibrationScheduler {
    async fn schedule_calibration(&self, device_id: &str, calibration_type: CalibrationType) -> QarResult<String>;
    async fn should_calibrate(&self, device_id: &str) -> QarResult<bool>;
    async fn get_calibration_schedule(&self, device_id: &str) -> QarResult<Vec<CalibrationTask>>;
}

/// Job requirements for device selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequirements {
    pub min_qubits: usize,
    pub max_shots: u64,
    pub required_gates: Vec<String>,
    pub max_queue_time_minutes: Option<u32>,
    pub min_fidelity: Option<f64>,
    pub prefer_hardware: bool,
    pub allow_simulation: bool,
}

/// Device health report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceHealthReport {
    pub device_id: String,
    pub overall_health: f64,
    pub status: DeviceStatus,
    pub issues: Vec<String>,
    pub recommendations: Vec<String>,
    pub last_calibration: Option<DateTime<Utc>>,
    pub next_maintenance: Option<DateTime<Utc>>,
}

/// Device metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetrics {
    pub device_id: String,
    pub timestamp: DateTime<Utc>,
    pub queue_length: u32,
    pub active_jobs: u32,
    pub average_fidelity: f64,
    pub error_rate: f64,
    pub uptime_percentage: f64,
    pub temperature: Option<f64>,
    pub performance_score: f64,
}

/// Device anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceAnomaly {
    pub device_id: String,
    pub anomaly_type: String,
    pub severity: AnomalySeverity,
    pub description: String,
    pub detected_at: DateTime<Utc>,
    pub metric_value: f64,
    pub expected_range: (f64, f64),
}

/// Anomaly severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Calibration types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationType {
    Full,
    GateCalibration,
    ReadoutCalibration,
    FrequencyCalibration,
    CrossTalkCalibration,
}

/// Calibration task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationTask {
    pub id: String,
    pub device_id: String,
    pub calibration_type: CalibrationType,
    pub scheduled_at: DateTime<Utc>,
    pub estimated_duration_minutes: u32,
    pub priority: u8,
}

impl QuantumDevice {
    /// Create new quantum device
    pub fn new(
        device_type: DeviceType,
        name: String,
        config: DeviceConfig,
        device_executor: Arc<dyn DeviceExecutor + Send + Sync>,
    ) -> Self {
        let capabilities = device_executor.get_capabilities();
        
        Self {
            device_type,
            name,
            config,
            capabilities,
            status: Arc::new(RwLock::new(DeviceStatus::Offline)),
            job_queue: Arc::new(RwLock::new(Vec::new())),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            completed_jobs: Arc::new(RwLock::new(Vec::new())),
            device_executor,
            calibration_data: Arc::new(RwLock::new(CalibrationData {
                calibrated_at: Utc::now(),
                gate_fidelities: HashMap::new(),
                qubit_frequencies: Vec::new(),
                coupling_strengths: HashMap::new(),
                readout_fidelities: Vec::new(),
                coherence_times: Vec::new(),
                cross_talk_matrix: Vec::new(),
                temperature_mk: None,
            })),
            usage_stats: Arc::new(Mutex::new(DeviceUsageStats {
                total_jobs: 0,
                successful_jobs: 0,
                failed_jobs: 0,
                total_shots: 0,
                total_execution_time_ms: 0,
                average_queue_time_ms: 0.0,
                average_execution_time_ms: 0.0,
                utilization_percentage: 0.0,
                last_updated: Utc::now(),
            })),
        }
    }

    /// Submit job to device
    pub async fn submit_job(&self, mut job: QuantumJob) -> QarResult<String> {
        job.id = Uuid::new_v4().to_string();
        job.submitted_at = Utc::now();
        job.status = JobStatus::Queued;

        // Validate job against device capabilities
        self.validate_job(&job).await?;

        // Add to queue
        {
            let mut queue = self.job_queue.write().await;
            queue.push(job.clone());
        }

        // Try to start execution if device is available
        self.try_start_next_job().await?;

        Ok(job.id)
    }

    /// Validate job against device capabilities
    async fn validate_job(&self, job: &QuantumJob) -> QarResult<()> {
        if job.circuit.num_qubits > self.capabilities.max_qubits {
            return Err(QarError::ValidationError(
                format!("Circuit requires {} qubits, device has {}", 
                       job.circuit.num_qubits, self.capabilities.max_qubits)
            ));
        }

        if job.shots > self.capabilities.max_shots {
            return Err(QarError::ValidationError(
                format!("Requested {} shots, device supports max {}", 
                       job.shots, self.capabilities.max_shots)
            ));
        }

        // Validate gate set
        for gate in &job.circuit.gates {
            if !self.capabilities.gate_set.contains(&gate.gate_type) {
                return Err(QarError::ValidationError(
                    format!("Gate {} not supported by device", gate.gate_type)
                ));
            }
        }

        Ok(())
    }

    /// Try to start next job from queue
    async fn try_start_next_job(&self) -> QarResult<()> {
        let status = self.status.read().await;
        if *status != DeviceStatus::Online {
            return Ok(());
        }
        drop(status);

        let active_count = {
            let active = self.active_jobs.read().await;
            active.len()
        };

        if active_count >= self.config.max_parallel_experiments as usize {
            return Ok(());
        }

        let next_job = {
            let mut queue = self.job_queue.write().await;
            queue.pop()
        };

        if let Some(job) = next_job {
            self.start_job_execution(job).await?;
        }

        Ok(())
    }

    /// Start job execution
    async fn start_job_execution(&self, mut job: QuantumJob) -> QarResult<()> {
        job.status = JobStatus::Running;
        job.started_at = Some(Utc::now());

        let job_id = job.id.clone();

        // Add to active jobs
        {
            let mut active = self.active_jobs.write().await;
            active.insert(job_id.clone(), job.clone());
        }

        // Execute in background
        let device_executor = self.device_executor.clone();
        let config = self.config.clone();
        let device_ref = self.clone_for_execution();

        tokio::spawn(async move {
            let result = device_ref.execute_job_async(job, device_executor, config).await;
            if let Err(e) = result {
                log::error!("Job execution failed: {}", e);
            }
        });

        Ok(())
    }

    /// Execute job asynchronously
    async fn execute_job_async(
        &self,
        mut job: QuantumJob,
        executor: Arc<dyn DeviceExecutor + Send + Sync>,
        config: DeviceConfig,
    ) -> QarResult<()> {
        let start_time = std::time::Instant::now();

        let result = executor.execute_circuit(&job.circuit, job.shots, &config).await;

        let execution_time = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(job_result) => {
                job.status = JobStatus::Completed;
                job.completed_at = Some(Utc::now());
                job.result = Some(job_result);
            }
            Err(e) => {
                job.status = JobStatus::Failed;
                job.completed_at = Some(Utc::now());
                job.error_message = Some(e.to_string());
            }
        }

        // Update job status
        self.complete_job(job, execution_time).await?;

        // Try to start next job
        self.try_start_next_job().await?;

        Ok(())
    }

    /// Complete job execution
    async fn complete_job(&self, job: QuantumJob, execution_time: u64) -> QarResult<()> {
        // Remove from active jobs
        {
            let mut active = self.active_jobs.write().await;
            active.remove(&job.id);
        }

        // Add to completed jobs
        {
            let mut completed = self.completed_jobs.write().await;
            completed.push(job.clone());

            // Limit completed jobs history
            if completed.len() > 10000 {
                completed.drain(0..1000);
            }
        }

        // Update usage statistics
        self.update_usage_stats(&job, execution_time).await?;

        Ok(())
    }

    /// Update device usage statistics
    async fn update_usage_stats(&self, job: &QuantumJob, execution_time: u64) -> QarResult<()> {
        let mut stats = self.usage_stats.lock().await;

        stats.total_jobs += 1;
        stats.total_shots += job.shots;
        stats.total_execution_time_ms += execution_time;

        if job.status == JobStatus::Completed {
            stats.successful_jobs += 1;
        } else if job.status == JobStatus::Failed {
            stats.failed_jobs += 1;
        }

        stats.average_execution_time_ms = 
            stats.total_execution_time_ms as f64 / stats.total_jobs as f64;

        if let (Some(submitted), Some(started)) = (&job.submitted_at, &job.started_at) {
            let queue_time = (*started - *submitted).num_milliseconds() as f64;
            stats.average_queue_time_ms = 
                (stats.average_queue_time_ms * (stats.total_jobs - 1) as f64 + queue_time) / stats.total_jobs as f64;
        }

        stats.last_updated = Utc::now();

        Ok(())
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: &str) -> QarResult<Option<QuantumJob>> {
        // Check active jobs
        {
            let active = self.active_jobs.read().await;
            if let Some(job) = active.get(job_id) {
                return Ok(Some(job.clone()));
            }
        }

        // Check completed jobs
        {
            let completed = self.completed_jobs.read().await;
            for job in completed.iter().rev() {
                if job.id == job_id {
                    return Ok(Some(job.clone()));
                }
            }
        }

        // Check queue
        {
            let queue = self.job_queue.read().await;
            for job in queue.iter() {
                if job.id == job_id {
                    return Ok(Some(job.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Cancel job
    pub async fn cancel_job(&self, job_id: &str) -> QarResult<()> {
        // Remove from queue
        {
            let mut queue = self.job_queue.write().await;
            queue.retain(|job| job.id != job_id);
        }

        // Update active job status
        {
            let mut active = self.active_jobs.write().await;
            if let Some(job) = active.get_mut(job_id) {
                job.status = JobStatus::Cancelled;
                job.completed_at = Some(Utc::now());
            }
        }

        Ok(())
    }

    /// Clone for async execution
    fn clone_for_execution(&self) -> Self {
        Self {
            device_type: self.device_type.clone(),
            name: self.name.clone(),
            config: self.config.clone(),
            capabilities: self.capabilities.clone(),
            status: self.status.clone(),
            job_queue: self.job_queue.clone(),
            active_jobs: self.active_jobs.clone(),
            completed_jobs: self.completed_jobs.clone(),
            device_executor: self.device_executor.clone(),
            calibration_data: self.calibration_data.clone(),
            usage_stats: self.usage_stats.clone(),
        }
    }

    /// Get device usage statistics
    pub async fn get_usage_stats(&self) -> QarResult<DeviceUsageStats> {
        let stats = self.usage_stats.lock().await;
        Ok(stats.clone())
    }

    /// Calibrate device
    pub async fn calibrate(&self, calibration_type: CalibrationType) -> QarResult<CalibrationData> {
        let calibration_data = self.device_executor.calibrate_device().await?;
        
        {
            let mut data = self.calibration_data.write().await;
            *data = calibration_data.clone();
        }

        Ok(calibration_data)
    }
}

impl QuantumHardwareManager {
    /// Create new hardware manager
    pub fn new(
        job_scheduler: Arc<dyn JobScheduler + Send + Sync>,
        monitoring_service: Arc<dyn DeviceMonitoringService + Send + Sync>,
        calibration_scheduler: Arc<dyn CalibrationScheduler + Send + Sync>,
    ) -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            device_registry: Arc::new(RwLock::new(HashMap::new())),
            job_scheduler,
            monitoring_service,
            calibration_scheduler,
        }
    }

    /// Register quantum device
    pub async fn register_device(&self, device: QuantumDevice) -> QarResult<()> {
        let device_id = device.name.clone();
        let device_type = device.device_type.clone();

        {
            let mut devices = self.devices.write().await;
            devices.insert(device_id.clone(), device);
        }

        {
            let mut registry = self.device_registry.write().await;
            registry.entry(device_type).or_insert_with(Vec::new).push(device_id);
        }

        Ok(())
    }

    /// Submit job with automatic device selection
    pub async fn submit_job(&self, circuit: QuantumCircuit, requirements: JobRequirements) -> QarResult<String> {
        let optimal_device = self.job_scheduler.get_optimal_device(&circuit, &requirements).await?;
        
        let job = QuantumJob {
            id: String::new(), // Will be set by device
            circuit,
            device_type: DeviceType::LightningQubit, // Will be updated
            shots: requirements.max_shots,
            metadata: HashMap::new(),
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            status: JobStatus::Queued,
            result: None,
            error_message: None,
            queue_position: None,
        };

        let devices = self.devices.read().await;
        let device = devices.get(&optimal_device)
            .ok_or_else(|| QarError::DeviceError(format!("Device not found: {}", optimal_device)))?;

        device.submit_job(job).await
    }

    /// Get available devices
    pub async fn get_available_devices(&self, device_type: Option<DeviceType>) -> QarResult<Vec<String>> {
        if let Some(dtype) = device_type {
            let registry = self.device_registry.read().await;
            Ok(registry.get(&dtype).cloned().unwrap_or_default())
        } else {
            let devices = self.devices.read().await;
            Ok(devices.keys().cloned().collect())
        }
    }

    /// Monitor device health
    pub async fn monitor_device_health(&self, device_id: &str) -> QarResult<DeviceHealthReport> {
        self.monitoring_service.monitor_device_health(device_id).await
    }
}

/// Mock implementations for testing
pub struct MockDeviceExecutor {
    pub capabilities: DeviceCapabilities,
}

impl MockDeviceExecutor {
    pub fn new() -> Self {
        Self {
            capabilities: DeviceCapabilities {
                max_qubits: 10,
                max_shots: 1000000,
                gate_set: vec!["X".to_string(), "Y".to_string(), "Z".to_string(), "H".to_string(), "CNOT".to_string()],
                connectivity: vec![(0, 1), (1, 2), (2, 3)],
                noise_model: None,
                coherence_time_us: Some(100.0),
                gate_fidelity: Some(0.99),
                readout_fidelity: Some(0.95),
                supports_measurements: true,
                supports_conditional: false,
                supports_reset: true,
            },
        }
    }
}

#[async_trait::async_trait]
impl DeviceExecutor for MockDeviceExecutor {
    async fn execute_circuit(&self, circuit: &QuantumCircuit, shots: u64, _config: &DeviceConfig) -> QarResult<JobResult> {
        // Simulate execution
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let mut counts = HashMap::new();
        counts.insert("0".repeat(circuit.num_qubits), shots / 2);
        counts.insert("1".repeat(circuit.num_qubits), shots / 2);

        Ok(JobResult {
            counts,
            probabilities: Some(vec![0.5; 2_usize.pow(circuit.num_qubits as u32)]),
            statevector: None,
            expectation_values: None,
            execution_time_ms: 100,
            shots_used: shots,
            fidelity: Some(0.98),
            metadata: HashMap::new(),
        })
    }

    async fn get_device_status(&self) -> QarResult<DeviceStatus> {
        Ok(DeviceStatus::Online)
    }

    async fn calibrate_device(&self) -> QarResult<CalibrationData> {
        Ok(CalibrationData {
            calibrated_at: Utc::now(),
            gate_fidelities: HashMap::new(),
            qubit_frequencies: vec![5.0; 10],
            coupling_strengths: HashMap::new(),
            readout_fidelities: vec![0.95; 10],
            coherence_times: vec![(100.0, 50.0); 10],
            cross_talk_matrix: Vec::new(),
            temperature_mk: Some(15.0),
        })
    }

    async fn estimate_execution_time(&self, _circuit: &QuantumCircuit, _shots: u64) -> QarResult<u64> {
        Ok(1000) // 1 second
    }

    fn get_capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> QuantumDevice {
        let config = DeviceConfig {
            device_type: DeviceType::LightningQubit,
            backend_name: None,
            shots: 1024,
            seed: Some(42),
            optimization_level: 1,
            initial_layout: None,
            coupling_map: None,
            basis_gates: None,
            noise_model: None,
            memory: false,
            max_parallel_experiments: 1,
            provider_config: HashMap::new(),
        };

        QuantumDevice::new(
            DeviceType::DefaultQubit,
            "test_device".to_string(),
            config,
            Arc::new(MockDeviceExecutor::new()),
        )
    }

    fn create_test_circuit() -> QuantumCircuit {
        let mut circuit = QuantumCircuit::new(2);
        circuit.h(0);
        circuit.cnot(0, 1);
        circuit
    }

    #[tokio::test]
    async fn test_device_creation() {
        let device = create_test_device();
        assert_eq!(device.name, "test_device");
        assert_eq!(device.device_type, DeviceType::DefaultQubit);
    }

    #[tokio::test]
    async fn test_job_submission() {
        let device = create_test_device();
        let circuit = create_test_circuit();

        let job = QuantumJob {
            id: String::new(),
            circuit,
            device_type: DeviceType::LightningQubit,
            shots: 1024,
            metadata: HashMap::new(),
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            status: JobStatus::Queued,
            result: None,
            error_message: None,
            queue_position: None,
        };

        let job_id = device.submit_job(job).await.unwrap();
        assert!(!job_id.is_empty());

        // Wait for job to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let job_status = device.get_job_status(&job_id).await.unwrap();
        assert!(job_status.is_some());
    }

    #[tokio::test]
    async fn test_job_validation() {
        let device = create_test_device();
        let mut circuit = create_test_circuit();
        
        // Create circuit with too many qubits
        circuit.num_qubits = 20;

        let job = QuantumJob {
            id: String::new(),
            circuit,
            device_type: DeviceType::LightningQubit,
            shots: 1024,
            metadata: HashMap::new(),
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            status: JobStatus::Queued,
            result: None,
            error_message: None,
            queue_position: None,
        };

        let result = device.submit_job(job).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_device_calibration() {
        let device = create_test_device();
        
        let calibration_data = device.calibrate(CalibrationType::Full).await.unwrap();
        assert!(!calibration_data.qubit_frequencies.is_empty());
        assert!(!calibration_data.readout_fidelities.is_empty());
    }

    #[tokio::test]
    async fn test_usage_statistics() {
        let device = create_test_device();
        let circuit = create_test_circuit();

        let job = QuantumJob {
            id: String::new(),
            circuit,
            device_type: DeviceType::LightningQubit,
            shots: 1024,
            metadata: HashMap::new(),
            submitted_at: Utc::now(),
            started_at: None,
            completed_at: None,
            status: JobStatus::Queued,
            result: None,
            error_message: None,
            queue_position: None,
        };

        let _job_id = device.submit_job(job).await.unwrap();
        
        // Wait for job to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let stats = device.get_usage_stats().await.unwrap();
        assert!(stats.total_jobs > 0);
    }
}