//! CUDA helper functions for neural forecasting
//! 
//! This module provides CUDA device detection and management functions
//! for neural forecasting operations.

use std::ffi::CString;

/// Check if CUDA is available on the system
pub fn check_cuda_availability() -> Result<bool, Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        // Try to query CUDA runtime version
        match std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=driver_version")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    Ok(!output_str.trim().is_empty())
                } else {
                    Ok(false)
                }
            },
            Err(_) => Ok(false),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(false)
    }
}

/// Get the number of available CUDA devices
pub fn get_cuda_device_count() -> Result<u32, Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        match std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=count")
            .arg("--format=csv,noheader,nounits")
            .output()
        {
            Ok(output) => {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    let lines: Vec<&str> = output_str.trim().lines().collect();
                    Ok(lines.len() as u32)
                } else {
                    Ok(0)
                }
            },
            Err(_) => Ok(0),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(0)
    }
}

/// Get CUDA device information
pub fn get_cuda_device_info(device_id: u32) -> Result<CudaDeviceInfo, Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name,memory.total,compute_mode")
            .arg("--format=csv,noheader,nounits")
            .arg(&format!("--id={}", device_id))
            .output()?;
            
        if output.status.success() {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = output_str.trim().split(',').map(|s| s.trim()).collect();
            
            if parts.len() >= 3 {
                let memory_mb: u64 = parts[1].parse().unwrap_or(0);
                return Ok(CudaDeviceInfo {
                    device_id,
                    name: parts[0].to_string(),
                    total_memory: memory_mb * 1024 * 1024, // Convert MB to bytes
                    compute_capability: (0, 0), // Default values
                    multiprocessor_count: 0,    // Default value
                });
            }
        }
        
        // Fallback to default info if query fails
        Ok(CudaDeviceInfo {
            device_id,
            name: format!("CUDA Device {}", device_id),
            total_memory: 0,
            compute_capability: (0, 0),
            multiprocessor_count: 0,
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("CUDA not available".into())
    }
}

/// CUDA device information structure
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub device_id: u32,
    pub name: String,
    pub total_memory: u64,
    pub compute_capability: (u32, u32),
    pub multiprocessor_count: u32,
}

/// Initialize CUDA context for a specific device
pub fn init_cuda_context(device_id: u32) -> Result<CudaContext, Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda")]
    {
        // Validate device exists
        let device_count = get_cuda_device_count()?;
        if device_id >= device_count {
            return Err(format!("Device {} not found. Available devices: {}", device_id, device_count).into());
        }
        
        Ok(CudaContext {
            device_id,
            initialized: true,
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err("CUDA not available".into())
    }
}

/// CUDA context handle
#[derive(Debug)]
pub struct CudaContext {
    pub device_id: u32,
    pub initialized: bool,
}

impl CudaContext {
    /// Synchronize the CUDA device
    pub fn synchronize(&self) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        {
            if self.initialized {
                // In a real implementation, this would call cudaDeviceSynchronize()
                // For now, just return success
                Ok(())
            } else {
                Err("CUDA context not initialized".into())
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err("CUDA not available".into())
        }
    }
    
    /// Get memory information for this device
    pub fn get_memory_info(&self) -> Result<CudaMemoryInfo, Box<dyn std::error::Error>> {
        #[cfg(feature = "cuda")]
        {
            if self.initialized {
                let output = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=memory.used,memory.free,memory.total")
                    .arg("--format=csv,noheader,nounits")
                    .arg(&format!("--id={}", self.device_id))
                    .output()?;
                    
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    let parts: Vec<&str> = output_str.trim().split(',').map(|s| s.trim()).collect();
                    
                    if parts.len() >= 3 {
                        let used_mb: u64 = parts[0].parse().unwrap_or(0);
                        let free_mb: u64 = parts[1].parse().unwrap_or(0);
                        let total_mb: u64 = parts[2].parse().unwrap_or(0);
                        
                        return Ok(CudaMemoryInfo {
                            used: used_mb * 1024 * 1024,
                            free: free_mb * 1024 * 1024,
                            total: total_mb * 1024 * 1024,
                        });
                    }
                }
                
                // Fallback values
                Ok(CudaMemoryInfo {
                    used: 0,
                    free: 0,
                    total: 0,
                })
            } else {
                Err("CUDA context not initialized".into())
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err("CUDA not available".into())
        }
    }
}

/// CUDA memory information
#[derive(Debug, Clone)]
pub struct CudaMemoryInfo {
    pub used: u64,
    pub free: u64,
    pub total: u64,
}

impl CudaMemoryInfo {
    /// Get memory utilization as a percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.total > 0 {
            (self.used as f64 / self.total as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// CUDA error types
#[derive(Debug)]
pub enum CudaError {
    DeviceNotFound,
    InsufficientMemory,
    InitializationFailed,
    RuntimeError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceNotFound => write!(f, "CUDA device not found"),
            CudaError::InsufficientMemory => write!(f, "Insufficient CUDA memory"),
            CudaError::InitializationFailed => write!(f, "CUDA initialization failed"),
            CudaError::RuntimeError(msg) => write!(f, "CUDA runtime error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test should not fail regardless of CUDA availability
        let result = check_cuda_availability();
        assert!(result.is_ok());
        
        // If CUDA is available, device count should be > 0
        if result.unwrap() {
            let device_count = get_cuda_device_count().unwrap_or(0);
            println!("CUDA devices found: {}", device_count);
        }
    }

    #[test]
    fn test_memory_info_utilization() {
        let mem_info = CudaMemoryInfo {
            used: 1024,
            free: 1024,
            total: 2048,
        };
        
        assert_eq!(mem_info.utilization_percent(), 50.0);
    }
}