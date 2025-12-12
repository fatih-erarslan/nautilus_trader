//! Hardware configuration detection and optimization
//!
//! This module provides automatic hardware detection and configuration optimization
//! for CDFA operations, including CPU features, GPU capabilities, and memory topology.

use crate::error::Result;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Hardware configuration information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HardwareConfig {
    /// CPU information
    pub cpu_vendor: String,
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub cpu_features: Vec<String>,
    pub cpu_cache_sizes: CpuCacheInfo,
    
    /// Memory information
    pub total_memory_gb: usize,
    pub available_memory_gb: usize,
    pub memory_speed_mhz: Option<u32>,
    pub numa_nodes: usize,
    
    /// GPU information
    pub has_gpu: bool,
    pub gpu_devices: Vec<u32>,
    pub gpu_memory_gb: Vec<usize>,
    pub gpu_compute_capabilities: Vec<String>,
    
    /// Storage information
    pub storage_type: StorageType,
    pub storage_speed_mb_s: Option<u32>,
    
    /// Operating system information
    pub os_type: String,
    pub os_version: String,
    pub kernel_version: String,
    
    /// Runtime environment
    pub is_container: bool,
    pub available_libraries: Vec<String>,
    
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// CPU cache information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CpuCacheInfo {
    pub l1_data_kb: Option<u32>,
    pub l1_instruction_kb: Option<u32>,
    pub l2_kb: Option<u32>,
    pub l3_kb: Option<u32>,
}

/// Storage type enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StorageType {
    Hdd,
    Ssd,
    Nvme,
    Network,
    Unknown,
}

/// Performance profile based on hardware capabilities
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceProfile {
    /// Overall performance tier (1=low, 2=medium, 3=high, 4=extreme)
    pub tier: u8,
    /// Recommended parallel thread count
    pub recommended_threads: usize,
    /// Recommended batch size for operations
    pub recommended_batch_size: usize,
    /// Recommended memory limit in MB
    pub recommended_memory_limit_mb: usize,
    /// SIMD optimization recommendations
    pub simd_recommendations: Vec<String>,
    /// GPU usage recommendations
    pub gpu_recommendations: GpuRecommendations,
}

/// GPU usage recommendations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GpuRecommendations {
    pub use_gpu: bool,
    pub preferred_device_id: Option<u32>,
    pub recommended_memory_fraction: f32,
    pub suitable_for_ml: bool,
    pub suitable_for_compute: bool,
}

impl HardwareConfig {
    /// Detect hardware configuration automatically
    pub fn detect() -> Result<Self> {
        let mut config = Self {
            cpu_vendor: String::new(),
            cpu_brand: String::new(),
            cpu_cores: 1,
            cpu_threads: 1,
            cpu_features: Vec::new(),
            cpu_cache_sizes: CpuCacheInfo::default(),
            total_memory_gb: 1,
            available_memory_gb: 1,
            memory_speed_mhz: None,
            numa_nodes: 1,
            has_gpu: false,
            gpu_devices: Vec::new(),
            gpu_memory_gb: Vec::new(),
            gpu_compute_capabilities: Vec::new(),
            storage_type: StorageType::Unknown,
            storage_speed_mb_s: None,
            os_type: String::new(),
            os_version: String::new(),
            kernel_version: String::new(),
            is_container: false,
            available_libraries: Vec::new(),
            performance_profile: PerformanceProfile::default(),
        };
        
        // Detect CPU information
        config.detect_cpu_info()?;
        
        // Detect memory information
        config.detect_memory_info()?;
        
        // Detect GPU information
        config.detect_gpu_info()?;
        
        // Detect storage information
        config.detect_storage_info()?;
        
        // Detect OS information
        config.detect_os_info()?;
        
        // Detect runtime environment
        config.detect_runtime_environment()?;
        
        // Generate performance profile
        config.generate_performance_profile();
        
        Ok(config)
    }
    
    /// Detect CPU information
    fn detect_cpu_info(&mut self) -> Result<()> {
        // Use num_cpus for basic CPU information
        #[cfg(feature = "parallel")]
        {
            if let Some(num_cpus) = num_cpus::get_physical().try_into().ok() {
                self.cpu_cores = num_cpus;
            }
            self.cpu_threads = num_cpus::get();
        }
        
        // Detect CPU features using raw-cpuid if available
        #[cfg(feature = "simd")]
        {
            self.detect_cpu_features()?;
        }
        
        // Fallback values if detection fails
        if self.cpu_cores == 0 {
            self.cpu_cores = 1;
        }
        if self.cpu_threads == 0 {
            self.cpu_threads = 1;
        }
        
        Ok(())
    }
    
    /// Detect CPU features
    #[cfg(all(feature = "simd", feature = "runtime-detection"))]
    fn detect_cpu_features(&mut self) -> Result<()> {
        use raw_cpuid::CpuId;
        
        let cpuid = CpuId::new();
        
        // Get vendor info
        if let Some(vendor) = cpuid.get_vendor_info() {
            self.cpu_vendor = vendor.as_str().to_string();
        }
        
        // Get brand string
        if let Some(brand) = cpuid.get_processor_brand_string() {
            self.cpu_brand = brand.as_str().to_string();
        }
        
        // Get feature information
        if let Some(features) = cpuid.get_feature_info() {
            let mut cpu_features = Vec::new();
            
            if features.has_sse() { cpu_features.push("sse".to_string()); }
            if features.has_sse2() { cpu_features.push("sse2".to_string()); }
            if features.has_sse3() { cpu_features.push("sse3".to_string()); }
            if features.has_ssse3() { cpu_features.push("ssse3".to_string()); }
            if features.has_sse41() { cpu_features.push("sse4.1".to_string()); }
            if features.has_sse42() { cpu_features.push("sse4.2".to_string()); }
            if features.has_avx() { cpu_features.push("avx".to_string()); }
            if features.has_fma() { cpu_features.push("fma".to_string()); }
            if features.has_aes() { cpu_features.push("aes".to_string()); }
            
            self.cpu_features = cpu_features;
        }
        
        // Get extended features
        if let Some(ext_features) = cpuid.get_extended_feature_info() {
            if ext_features.has_avx2() { 
                self.cpu_features.push("avx2".to_string()); 
            }
            if ext_features.has_avx512f() { 
                self.cpu_features.push("avx512f".to_string()); 
            }
            if ext_features.has_avx512cd() { 
                self.cpu_features.push("avx512cd".to_string()); 
            }
            if ext_features.has_avx512er() { 
                self.cpu_features.push("avx512er".to_string()); 
            }
            if ext_features.has_avx512pf() { 
                self.cpu_features.push("avx512pf".to_string()); 
            }
        }
        
        // Get cache information
        if let Some(cache_info) = cpuid.get_cache_info() {
            for cache in cache_info {
                match cache.level {
                    #[cfg(feature = "runtime-detection")]
                    1 if cache.cache_type == raw_cpuid::CacheType::Data => {
                        self.cpu_cache_sizes.l1_data_kb = Some(cache.cache_size / 1024);
                    },
                    1 if cache.cache_type == raw_cpuid::CacheType::Instruction => {
                        self.cpu_cache_sizes.l1_instruction_kb = Some(cache.cache_size / 1024);
                    },
                    2 => {
                        self.cpu_cache_sizes.l2_kb = Some(cache.cache_size / 1024);
                    },
                    3 => {
                        self.cpu_cache_sizes.l3_kb = Some(cache.cache_size / 1024);
                    },
                    _ => {},
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect memory information
    fn detect_memory_info(&mut self) -> Result<()> {
        // Basic memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(size_str) = line.split_whitespace().nth(1) {
                            if let Ok(size_kb) = size_str.parse::<usize>() {
                                self.total_memory_gb = size_kb / 1024 / 1024;
                            }
                        }
                    }
                    if line.starts_with("MemAvailable:") {
                        if let Some(size_str) = line.split_whitespace().nth(1) {
                            if let Ok(size_kb) = size_str.parse::<usize>() {
                                self.available_memory_gb = size_kb / 1024 / 1024;
                            }
                        }
                    }
                }
            }
        }
        
        // Fallback to system-specific methods or defaults
        if self.total_memory_gb == 0 {
            self.total_memory_gb = 4; // 4GB default
            self.available_memory_gb = 3; // Conservative estimate
        }
        
        Ok(())
    }
    
    /// Detect GPU information
    fn detect_gpu_info(&mut self) -> Result<()> {
        // GPU detection would typically use libraries like CUDA runtime or OpenCL
        // For now, we'll provide a basic implementation
        
        #[cfg(feature = "gpu")]
        {
            // Try to detect NVIDIA GPUs
            if std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=count")
                .arg("--format=csv,noheader,nounits")
                .output()
                .is_ok()
            {
                self.has_gpu = true;
                // In a real implementation, parse nvidia-smi output
                self.gpu_devices = vec![0]; // Assume one GPU for now
                self.gpu_memory_gb = vec![8]; // Assume 8GB for now
                self.gpu_compute_capabilities = vec!["8.6".to_string()]; // Modern capability
            }
        }
        
        Ok(())
    }
    
    /// Detect storage information
    fn detect_storage_info(&mut self) -> Result<()> {
        // Storage detection is OS-specific
        #[cfg(target_os = "linux")]
        {
            // Check if root filesystem is on SSD
            if let Ok(output) = std::process::Command::new("lsblk")
                .arg("-d")
                .arg("-o")
                .arg("name,rota")
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines().skip(1) {
                    if let Some(rota) = line.split_whitespace().nth(1) {
                        if rota == "0" {
                            self.storage_type = StorageType::Ssd;
                            break;
                        } else if rota == "1" {
                            self.storage_type = StorageType::Hdd;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect OS information
    fn detect_os_info(&mut self) -> Result<()> {
        self.os_type = std::env::consts::OS.to_string();
        
        #[cfg(target_os = "linux")]
        {
            if let Ok(release) = std::fs::read_to_string("/etc/os-release") {
                for line in release.lines() {
                    if line.starts_with("VERSION=") {
                        self.os_version = line.split('=').nth(1)
                            .unwrap_or("")
                            .trim_matches('"')
                            .to_string();
                        break;
                    }
                }
            }
            
            if let Ok(version) = std::fs::read_to_string("/proc/version") {
                if let Some(kernel_part) = version.split_whitespace().nth(2) {
                    self.kernel_version = kernel_part.to_string();
                }
            }
        }
        
        Ok(())
    }
    
    /// Detect runtime environment
    fn detect_runtime_environment(&mut self) -> Result<()> {
        // Check if running in container
        #[cfg(target_os = "linux")]
        {
            self.is_container = std::path::Path::new("/.dockerenv").exists() ||
                std::fs::read_to_string("/proc/1/cgroup")
                    .map(|s| s.contains("docker") || s.contains("containerd"))
                    .unwrap_or(false);
        }
        
        // Detect available libraries
        let mut libraries = Vec::new();
        
        // Check for BLAS libraries
        if std::process::Command::new("ldconfig")
            .arg("-p")
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).contains("blas"))
            .unwrap_or(false)
        {
            libraries.push("blas".to_string());
        }
        
        // Check for Intel MKL
        if std::env::var("MKLROOT").is_ok() {
            libraries.push("mkl".to_string());
        }
        
        // Check for OpenMP
        if std::env::var("OMP_NUM_THREADS").is_ok() {
            libraries.push("openmp".to_string());
        }
        
        self.available_libraries = libraries;
        
        Ok(())
    }
    
    /// Generate performance profile based on detected hardware
    fn generate_performance_profile(&mut self) {
        let mut tier = 1u8;
        let mut recommended_threads = self.cpu_threads;
        let mut recommended_batch_size = 100;
        let mut recommended_memory_limit_mb = self.available_memory_gb * 512; // 50% of available
        let mut simd_recommendations = Vec::new();
        
        // Determine performance tier
        if self.cpu_cores >= 16 && self.total_memory_gb >= 32 {
            tier = 4; // Extreme performance
            recommended_batch_size = 10000;
        } else if self.cpu_cores >= 8 && self.total_memory_gb >= 16 {
            tier = 3; // High performance
            recommended_batch_size = 5000;
        } else if self.cpu_cores >= 4 && self.total_memory_gb >= 8 {
            tier = 2; // Medium performance
            recommended_batch_size = 1000;
        }
        
        // Adjust thread count based on workload type
        recommended_threads = std::cmp::min(recommended_threads, self.cpu_cores * 2);
        
        // SIMD recommendations based on CPU features
        if self.cpu_features.contains(&"avx512f".to_string()) {
            simd_recommendations.push("avx512".to_string());
        } else if self.cpu_features.contains(&"avx2".to_string()) {
            simd_recommendations.push("avx2".to_string());
        } else if self.cpu_features.contains(&"avx".to_string()) {
            simd_recommendations.push("avx".to_string());
        } else if self.cpu_features.contains(&"sse2".to_string()) {
            simd_recommendations.push("sse2".to_string());
        }
        
        // GPU recommendations
        let gpu_recommendations = GpuRecommendations {
            use_gpu: self.has_gpu,
            preferred_device_id: if self.has_gpu { Some(0) } else { None },
            recommended_memory_fraction: if self.has_gpu { 0.8 } else { 0.0 },
            suitable_for_ml: self.has_gpu && !self.gpu_memory_gb.is_empty() && self.gpu_memory_gb[0] >= 4,
            suitable_for_compute: self.has_gpu,
        };
        
        self.performance_profile = PerformanceProfile {
            tier,
            recommended_threads,
            recommended_batch_size,
            recommended_memory_limit_mb,
            simd_recommendations,
            gpu_recommendations,
        };
    }
    
    /// Get optimization recommendations as key-value pairs
    pub fn get_optimization_recommendations(&self) -> HashMap<String, String> {
        let mut recommendations = HashMap::new();
        
        recommendations.insert(
            "threads".to_string(),
            self.performance_profile.recommended_threads.to_string(),
        );
        recommendations.insert(
            "batch_size".to_string(),
            self.performance_profile.recommended_batch_size.to_string(),
        );
        recommendations.insert(
            "memory_limit_mb".to_string(),
            self.performance_profile.recommended_memory_limit_mb.to_string(),
        );
        recommendations.insert(
            "use_gpu".to_string(),
            self.performance_profile.gpu_recommendations.use_gpu.to_string(),
        );
        recommendations.insert(
            "performance_tier".to_string(),
            self.performance_profile.tier.to_string(),
        );
        
        if !self.performance_profile.simd_recommendations.is_empty() {
            recommendations.insert(
                "simd_features".to_string(),
                self.performance_profile.simd_recommendations.join(","),
            );
        }
        
        recommendations
    }
}

impl Default for CpuCacheInfo {
    fn default() -> Self {
        Self {
            l1_data_kb: None,
            l1_instruction_kb: None,
            l2_kb: None,
            l3_kb: None,
        }
    }
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self {
            tier: 1,
            recommended_threads: 1,
            recommended_batch_size: 100,
            recommended_memory_limit_mb: 1024,
            simd_recommendations: Vec::new(),
            gpu_recommendations: GpuRecommendations::default(),
        }
    }
}

impl Default for GpuRecommendations {
    fn default() -> Self {
        Self {
            use_gpu: false,
            preferred_device_id: None,
            recommended_memory_fraction: 0.0,
            suitable_for_ml: false,
            suitable_for_compute: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hardware_config_creation() {
        let config = HardwareConfig::detect();
        assert!(config.is_ok());
        
        let config = config.unwrap();
        assert!(config.cpu_cores > 0);
        assert!(config.cpu_threads > 0);
        assert!(config.total_memory_gb > 0);
    }
    
    #[test]
    fn test_performance_profile() {
        let mut config = HardwareConfig {
            cpu_cores: 8,
            cpu_threads: 16,
            total_memory_gb: 16,
            available_memory_gb: 12,
            cpu_features: vec!["avx2".to_string()],
            has_gpu: true,
            gpu_memory_gb: vec![8],
            ..Default::default()
        };
        
        config.generate_performance_profile();
        
        assert_eq!(config.performance_profile.tier, 3);
        assert!(config.performance_profile.gpu_recommendations.use_gpu);
        assert!(config.performance_profile.simd_recommendations.contains(&"avx2".to_string()));
    }
    
    #[test]
    fn test_optimization_recommendations() {
        let config = HardwareConfig {
            performance_profile: PerformanceProfile {
                tier: 2,
                recommended_threads: 4,
                recommended_batch_size: 1000,
                recommended_memory_limit_mb: 2048,
                simd_recommendations: vec!["avx".to_string()],
                gpu_recommendations: GpuRecommendations {
                    use_gpu: false,
                    ..Default::default()
                },
            },
            ..Default::default()
        };
        
        let recommendations = config.get_optimization_recommendations();
        assert_eq!(recommendations.get("threads"), Some(&"4".to_string()));
        assert_eq!(recommendations.get("batch_size"), Some(&"1000".to_string()));
        assert_eq!(recommendations.get("use_gpu"), Some(&"false".to_string()));
        assert_eq!(recommendations.get("performance_tier"), Some(&"2".to_string()));
    }
}

// Default implementation for HardwareConfig
impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            cpu_vendor: "Unknown".to_string(),
            cpu_brand: "Unknown".to_string(),
            cpu_cores: 1,
            cpu_threads: 1,
            cpu_features: Vec::new(),
            cpu_cache_sizes: CpuCacheInfo::default(),
            total_memory_gb: 4,
            available_memory_gb: 3,
            memory_speed_mhz: None,
            numa_nodes: 1,
            has_gpu: false,
            gpu_devices: Vec::new(),
            gpu_memory_gb: Vec::new(),
            gpu_compute_capabilities: Vec::new(),
            storage_type: StorageType::Unknown,
            storage_speed_mb_s: None,
            os_type: std::env::consts::OS.to_string(),
            os_version: "Unknown".to_string(),
            kernel_version: "Unknown".to_string(),
            is_container: false,
            available_libraries: Vec::new(),
            performance_profile: PerformanceProfile::default(),
        }
    }
}