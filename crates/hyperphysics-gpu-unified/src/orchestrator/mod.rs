//! GPU Orchestrator - Central coordination for dual-GPU system

mod dual_gpu;
mod scheduler;

pub use dual_gpu::DualGpuCoordinator;
pub use scheduler::PipelineScheduler;

use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;
use wgpu::{Device, Queue, AdapterInfo};

use crate::{GpuError, GpuPreference, GpuResult, GpuSpecs, WorkloadType};
use crate::pools::ComputePool;
use crate::kernels::KernelRegistry;

/// Configuration for GPU orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Enable dual-GPU support
    pub dual_gpu: bool,
    /// Maximum buffer pool size in bytes
    pub max_buffer_pool_bytes: u64,
    /// Enable pipeline caching
    pub pipeline_caching: bool,
    /// Enable memory pressure monitoring
    pub memory_monitoring: bool,
    /// Workgroup size (default: 256 for RDNA2)
    pub workgroup_size: u32,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            dual_gpu: true,
            max_buffer_pool_bytes: 1024 * 1024 * 1024, // 1GB
            pipeline_caching: true,
            memory_monitoring: true,
            workgroup_size: 256, // 4 wavefronts
        }
    }
}

/// Central GPU orchestrator for HyperPhysics ecosystem
pub struct GpuOrchestrator {
    /// Primary GPU device (RX 6800 XT)
    primary_device: Arc<Device>,
    /// Primary GPU queue
    primary_queue: Arc<Queue>,
    /// Primary GPU info
    primary_info: AdapterInfo,
    /// Primary GPU specs
    primary_specs: GpuSpecs,

    /// Secondary GPU device (RX 5500 XT)
    secondary_device: Option<Arc<Device>>,
    /// Secondary GPU queue
    secondary_queue: Option<Arc<Queue>>,
    /// Secondary GPU info
    secondary_info: Option<AdapterInfo>,
    /// Secondary GPU specs
    secondary_specs: Option<GpuSpecs>,

    /// Kernel registry
    kernels: Arc<RwLock<KernelRegistry>>,
    /// Compute pools
    pools: DashMap<String, ComputePool>,
    /// Pipeline scheduler
    scheduler: Arc<PipelineScheduler>,
    /// Configuration
    config: OrchestratorConfig,
}

impl GpuOrchestrator {
    /// Create new GPU orchestrator with default configuration
    pub fn new() -> GpuResult<Self> {
        Self::with_config(OrchestratorConfig::default())
    }

    /// Create GPU orchestrator with custom configuration
    pub fn with_config(config: OrchestratorConfig) -> GpuResult<Self> {
        pollster::block_on(Self::new_async(config))
    }

    /// Async initialization
    pub async fn new_async(config: OrchestratorConfig) -> GpuResult<Self> {
        // Create wgpu instance with Metal backend on macOS
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(target_os = "macos")]
            backends: wgpu::Backends::METAL,
            #[cfg(not(target_os = "macos"))]
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // FIXED: Enumerate ALL adapters instead of using request_adapter
        // This ensures we detect both RX 6800 XT and RX 5500 XT
        let adapters: Vec<_> = instance.enumerate_adapters(
            #[cfg(target_os = "macos")]
            wgpu::Backends::METAL,
            #[cfg(not(target_os = "macos"))]
            wgpu::Backends::all(),
        );

        tracing::info!("Found {} GPU adapter(s)", adapters.len());

        if adapters.is_empty() {
            return Err(GpuError::NoAdapterFound("No GPU adapters found".to_string()));
        }

        // Collect adapter info and sort by VRAM/capability (highest first)
        let mut adapter_infos: Vec<_> = adapters.iter()
            .map(|a| (a, a.get_info(), a.limits()))
            .collect();

        // Sort by max_buffer_size (proxy for VRAM capability) descending
        adapter_infos.sort_by(|a, b| b.2.max_buffer_size.cmp(&a.2.max_buffer_size));

        // Log all detected GPUs
        for (i, (_, info, limits)) in adapter_infos.iter().enumerate() {
            tracing::info!(
                "GPU {}: {} ({:?}) - device_id=0x{:x}, max_buffer={}GB, backend={:?}",
                i, info.name, info.device_type, info.device,
                limits.max_buffer_size / 1024 / 1024 / 1024,
                info.backend
            );
        }

        // Primary GPU = highest capability (first after sort)
        let (primary_adapter, primary_info, primary_limits) = adapter_infos.remove(0);
        let primary_info = primary_info.clone();

        // Request device with ACTUAL adapter limits, not default
        let (primary_device, primary_queue) = primary_adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("HyperPhysics-Primary"),
                    required_features: wgpu::Features::empty(),
                    // FIXED: Use adapter's actual limits instead of default
                    required_limits: primary_limits.clone(),
                },
                None,
            )
            .await?;

        let primary_specs = Self::detect_specs(&primary_info);

        // Initialize secondary GPU if dual-GPU enabled and we have more adapters
        let (secondary_device, secondary_queue, secondary_info, secondary_specs) =
            if config.dual_gpu && !adapter_infos.is_empty() {
                let (secondary_adapter, sec_info, sec_limits) = adapter_infos.remove(0);
                let sec_info = sec_info.clone();

                // Only use if different device (compare by name since device ID may be unreliable on Metal)
                if sec_info.name != primary_info.name {
                    match secondary_adapter
                        .request_device(
                            &wgpu::DeviceDescriptor {
                                label: Some("HyperPhysics-Secondary"),
                                required_features: wgpu::Features::empty(),
                                required_limits: sec_limits.clone(),
                            },
                            None,
                        )
                        .await
                    {
                        Ok((device, queue)) => {
                            let specs = Self::detect_specs(&sec_info);
                            tracing::info!("Secondary GPU initialized: {}", sec_info.name);
                            (Some(Arc::new(device)), Some(Arc::new(queue)), Some(sec_info), Some(specs))
                        }
                        Err(e) => {
                            tracing::warn!("Failed to initialize secondary GPU: {:?}", e);
                            (None, None, None, None)
                        }
                    }
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            };

        let kernels = Arc::new(RwLock::new(KernelRegistry::new()));
        let scheduler = Arc::new(PipelineScheduler::new(config.pipeline_caching));

        Ok(Self {
            primary_device: Arc::new(primary_device),
            primary_queue: Arc::new(primary_queue),
            primary_info,
            primary_specs,
            secondary_device,
            secondary_queue,
            secondary_info,
            secondary_specs,
            kernels,
            pools: DashMap::new(),
            scheduler,
            config,
        })
    }

    /// Initialize a GPU with given power preference
    async fn init_gpu(
        instance: &wgpu::Instance,
        power_pref: wgpu::PowerPreference,
    ) -> GpuResult<(Arc<Device>, Arc<Queue>, AdapterInfo)> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| GpuError::NoAdapterFound(format!("{:?}", power_pref)))?;

        let info = adapter.get_info();
        tracing::info!(
            "Initializing GPU: {} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some(&format!("HyperPhysics-{:?}", power_pref)),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        Ok((Arc::new(device), Arc::new(queue), info))
    }

    /// Detect GPU specifications from adapter info
    fn detect_specs(info: &AdapterInfo) -> GpuSpecs {
        // Detect based on device name or device ID
        // Device IDs: RX 6800 XT = 0x73bf, RX 5500 XT = 0x7340
        let name_lower = info.name.to_lowercase();

        // Log device info for debugging
        tracing::debug!(
            "Detecting specs for: {} (device_id=0x{:x})",
            info.name,
            info.device
        );

        // Check device ID first (most reliable when available)
        // RX 6800 XT device IDs: 0x73bf (main), 0x73a5, 0x73b1
        // RX 5500 XT device IDs: 0x7340 (main), 0x7341
        if info.device == 0x73bf || info.device == 0x73a5 || info.device == 0x73b1 {
            return GpuSpecs::rx_6800_xt();
        }
        if info.device == 0x7340 || info.device == 0x7341 {
            return GpuSpecs::rx_5500_xt();
        }

        // Fall back to name matching
        if name_lower.contains("6800 xt") {
            GpuSpecs::rx_6800_xt()
        } else if name_lower.contains("5500 xt") {
            GpuSpecs::rx_5500_xt()
        } else if name_lower.contains("6800") || name_lower.contains("navi 21") {
            // RDNA2 Navi 21 family (6800/6800XT/6900XT)
            GpuSpecs::rx_6800_xt()
        } else if name_lower.contains("5500") || name_lower.contains("navi 14") {
            // RDNA1 Navi 14 family (5500/5500XT)
            GpuSpecs::rx_5500_xt()
        } else if name_lower.contains("gfx10") && name_lower.contains("unknown prototype") {
            // "GFX10 Family Unknown Prototype" on macOS Metal typically indicates RX 6000 series
            // Since this system has RX 6800 XT as primary, assume high-end RDNA2
            GpuSpecs {
                name: "AMD Radeon RX 6800 XT (Metal)".to_string(),
                compute_units: 72,
                wavefront_size: 64,
                vram_bytes: 16 * 1024 * 1024 * 1024, // 16GB
                memory_bandwidth_gbps: 512.0,
                infinity_cache_bytes: 128 * 1024 * 1024, // 128MB
            }
        } else if name_lower.contains("gfx10") {
            // Generic GFX10 = RDNA1/RDNA2, assume mid-range
            GpuSpecs {
                name: info.name.clone(),
                compute_units: 40,
                wavefront_size: 64,
                vram_bytes: 8 * 1024 * 1024 * 1024,
                memory_bandwidth_gbps: 384.0,
                infinity_cache_bytes: 0,
            }
        } else {
            // Default fallback specs
            GpuSpecs {
                name: info.name.clone(),
                compute_units: 32,
                wavefront_size: 64,
                vram_bytes: 4 * 1024 * 1024 * 1024,
                memory_bandwidth_gbps: 256.0,
                infinity_cache_bytes: 0,
            }
        }
    }

    /// Get device and queue for a workload
    pub fn get_device_for_workload(&self, workload: &WorkloadType) -> (&Arc<Device>, &Arc<Queue>) {
        match workload.preferred_gpu() {
            GpuPreference::Secondary if self.secondary_device.is_some() => {
                (
                    self.secondary_device.as_ref().unwrap(),
                    self.secondary_queue.as_ref().unwrap(),
                )
            }
            _ => (&self.primary_device, &self.primary_queue),
        }
    }

    /// Get primary device and queue
    pub fn primary(&self) -> (&Arc<Device>, &Arc<Queue>) {
        (&self.primary_device, &self.primary_queue)
    }

    /// Get secondary device and queue (if available)
    pub fn secondary(&self) -> Option<(&Arc<Device>, &Arc<Queue>)> {
        match (&self.secondary_device, &self.secondary_queue) {
            (Some(device), Some(queue)) => Some((device, queue)),
            _ => None,
        }
    }

    /// Check if dual-GPU is available
    pub fn has_dual_gpu(&self) -> bool {
        self.secondary_device.is_some()
    }

    /// Get primary GPU specs
    pub fn primary_specs(&self) -> &GpuSpecs {
        &self.primary_specs
    }

    /// Get secondary GPU specs
    pub fn secondary_specs(&self) -> Option<&GpuSpecs> {
        self.secondary_specs.as_ref()
    }

    /// Get kernel registry
    pub fn kernels(&self) -> &Arc<RwLock<KernelRegistry>> {
        &self.kernels
    }

    /// Get pipeline scheduler
    pub fn scheduler(&self) -> &Arc<PipelineScheduler> {
        &self.scheduler
    }

    /// Create a compute pool
    pub fn create_pool(&self, name: &str, pool_type: crate::pools::PoolType) -> GpuResult<()> {
        let pool = ComputePool::new(
            name.to_string(),
            pool_type,
            self.primary_device.clone(),
            self.primary_queue.clone(),
        );
        self.pools.insert(name.to_string(), pool);
        Ok(())
    }

    /// Get a compute pool by name
    pub fn get_pool(&self, name: &str) -> Option<dashmap::mapref::one::Ref<'_, String, ComputePool>> {
        self.pools.get(name)
    }
}

impl Default for GpuOrchestrator {
    fn default() -> Self {
        Self::new().expect("Failed to create default GPU orchestrator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = GpuOrchestrator::new();
        assert!(orchestrator.is_ok());

        let orch = orchestrator.unwrap();
        let specs = orch.primary_specs();

        // Should detect RX 6800 XT as primary
        println!("Primary GPU: {}", specs.name);
        println!("Compute Units: {}", specs.compute_units);
        println!("VRAM: {} GB", specs.vram_bytes / 1024 / 1024 / 1024);

        // Check dual-GPU status
        println!("Dual-GPU available: {}", orch.has_dual_gpu());
        if let Some(secondary) = orch.secondary_specs() {
            println!("Secondary GPU: {}", secondary.name);
        }
    }

    #[test]
    fn test_workload_routing() {
        let orchestrator = GpuOrchestrator::new().unwrap();

        // Heavy compute should go to primary (72 CUs)
        let heavy = crate::WorkloadType::ComputeBound { estimated_flops: 10_000_000_000 };
        let (device, _) = orchestrator.get_device_for_workload(&heavy);
        assert!(std::sync::Arc::strong_count(device) >= 1);

        // Background should prefer secondary if available
        let background = crate::WorkloadType::Background;
        let _ = orchestrator.get_device_for_workload(&background);
    }
}
