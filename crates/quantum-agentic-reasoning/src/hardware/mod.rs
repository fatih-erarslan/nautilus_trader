//! Hardware Module
//!
//! Quantum hardware abstraction and device management for QAR trading operations.
//! Provides PennyLane-compatible device implementations and execution infrastructure.

// Core hardware modules
pub mod quantum_hardware;
pub mod devices;

// Re-export key types for convenience
pub use quantum_hardware::{
    QuantumHardwareManager, QuantumDevice, DeviceExecutor, DeviceCapabilities,
    DeviceConfig, DeviceType, DeviceStatus, QuantumJob, JobResult, JobStatus,
    JobRequirements, CalibrationData, CalibrationTask, CalibrationType,
    DeviceHealthReport, DeviceMetrics, DeviceAnomaly, AnomalySeverity,
    NoiseModel, ThermalRelaxation, DeviceUsageStats,
    JobScheduler, DeviceMonitoringService, CalibrationScheduler,
    MockDeviceExecutor
};

pub use devices::{
    LightningGpu, LightningKokkos, LightningQubit,
    select_optimal_lightning_device
};

// Backward compatibility aliases
pub type HardwareManager = QuantumHardwareManager;
pub type ResourceHandle = QuantumDevice;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that all key types are accessible
        let device_type = DeviceType::LightningQubit;
        assert_eq!(device_type, DeviceType::LightningQubit);
        
        let status = DeviceStatus::Online;
        assert_eq!(status, DeviceStatus::Online);
    }
    
    #[test]
    fn test_device_selection() {
        // Test device selection utility
        let selected = select_optimal_lightning_device(10, 50);
        assert_eq!(selected, "lightning.qubit");
        
        let selected = select_optimal_lightning_device(25, 300);
        assert_eq!(selected, "lightning.kokkos");
        
        let selected = select_optimal_lightning_device(35, 1000);
        assert_eq!(selected, "lightning.gpu");
    }
}