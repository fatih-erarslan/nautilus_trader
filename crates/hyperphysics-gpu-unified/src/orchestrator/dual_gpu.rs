//! Dual-GPU coordination for parallel workload distribution

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use wgpu::{Device, Queue};

use crate::{GpuPreference, GpuResult, WorkloadType};

/// Coordinator for dual-GPU workload distribution
pub struct DualGpuCoordinator {
    /// Primary GPU device (RX 6800 XT)
    primary_device: Arc<Device>,
    primary_queue: Arc<Queue>,

    /// Secondary GPU device (RX 5500 XT)
    secondary_device: Option<Arc<Device>>,
    secondary_queue: Option<Arc<Queue>>,

    /// Workload tracking
    primary_queue_depth: AtomicUsize,
    secondary_queue_depth: AtomicUsize,

    /// Memory pressure tracking
    primary_vram_used: AtomicU64,
    secondary_vram_used: AtomicU64,

    /// VRAM limits
    primary_vram_limit: u64,
    secondary_vram_limit: u64,
}

impl DualGpuCoordinator {
    /// Create new dual-GPU coordinator
    pub fn new(
        primary_device: Arc<Device>,
        primary_queue: Arc<Queue>,
        secondary_device: Option<Arc<Device>>,
        secondary_queue: Option<Arc<Queue>>,
    ) -> Self {
        Self {
            primary_device,
            primary_queue,
            secondary_device,
            secondary_queue,
            primary_queue_depth: AtomicUsize::new(0),
            secondary_queue_depth: AtomicUsize::new(0),
            primary_vram_used: AtomicU64::new(0),
            secondary_vram_used: AtomicU64::new(0),
            primary_vram_limit: 16 * 1024 * 1024 * 1024, // 16GB
            secondary_vram_limit: 4 * 1024 * 1024 * 1024, // 4GB
        }
    }

    /// Route workload to optimal GPU
    pub fn route(&self, workload: &WorkloadType) -> GpuAssignment {
        match workload {
            // Memory-bound: check VRAM availability
            WorkloadType::MemoryBound { required_vram } => {
                let primary_free = self.primary_vram_limit
                    .saturating_sub(self.primary_vram_used.load(Ordering::Relaxed));

                if *required_vram <= primary_free {
                    GpuAssignment::Primary
                } else if let Some(_) = &self.secondary_device {
                    let secondary_free = self.secondary_vram_limit
                        .saturating_sub(self.secondary_vram_used.load(Ordering::Relaxed));
                    if *required_vram <= secondary_free {
                        GpuAssignment::Secondary
                    } else {
                        GpuAssignment::Primary // Fallback
                    }
                } else {
                    GpuAssignment::Primary
                }
            }

            // Compute-bound: use high-CU GPU
            WorkloadType::ComputeBound { estimated_flops } => {
                if *estimated_flops > 1_000_000_000 {
                    GpuAssignment::Primary // 72 CUs
                } else if self.secondary_device.is_some() {
                    GpuAssignment::Secondary // 22 CUs sufficient
                } else {
                    GpuAssignment::Primary
                }
            }

            // Latency-critical: use less-loaded GPU
            WorkloadType::LatencyCritical { .. } => {
                let primary_depth = self.primary_queue_depth.load(Ordering::Relaxed);
                let secondary_depth = self.secondary_queue_depth.load(Ordering::Relaxed);

                if self.secondary_device.is_some() && secondary_depth < primary_depth {
                    GpuAssignment::Secondary
                } else {
                    GpuAssignment::Primary
                }
            }

            // Background: prefer secondary
            WorkloadType::Background => {
                if self.secondary_device.is_some() {
                    GpuAssignment::Secondary
                } else {
                    GpuAssignment::Primary
                }
            }
        }
    }

    /// Get device and queue for assignment
    pub fn get_resources(&self, assignment: GpuAssignment) -> (&Arc<Device>, &Arc<Queue>) {
        match assignment {
            GpuAssignment::Secondary if self.secondary_device.is_some() => {
                (
                    self.secondary_device.as_ref().unwrap(),
                    self.secondary_queue.as_ref().unwrap(),
                )
            }
            _ => (&self.primary_device, &self.primary_queue),
        }
    }

    /// Record workload submission
    pub fn submit_workload(&self, assignment: GpuAssignment, vram_bytes: u64) {
        match assignment {
            GpuAssignment::Primary => {
                self.primary_queue_depth.fetch_add(1, Ordering::Relaxed);
                self.primary_vram_used.fetch_add(vram_bytes, Ordering::Relaxed);
            }
            GpuAssignment::Secondary => {
                self.secondary_queue_depth.fetch_add(1, Ordering::Relaxed);
                self.secondary_vram_used.fetch_add(vram_bytes, Ordering::Relaxed);
            }
        }
    }

    /// Record workload completion
    pub fn complete_workload(&self, assignment: GpuAssignment, vram_bytes: u64) {
        match assignment {
            GpuAssignment::Primary => {
                self.primary_queue_depth.fetch_sub(1, Ordering::Relaxed);
                self.primary_vram_used.fetch_sub(vram_bytes, Ordering::Relaxed);
            }
            GpuAssignment::Secondary => {
                self.secondary_queue_depth.fetch_sub(1, Ordering::Relaxed);
                self.secondary_vram_used.fetch_sub(vram_bytes, Ordering::Relaxed);
            }
        }
    }

    /// Execute workloads on both GPUs in parallel
    #[cfg(feature = "async")]
    pub async fn parallel_execute<T, F1, F2>(
        &self,
        primary_task: F1,
        secondary_task: F2,
    ) -> GpuResult<(T, T)>
    where
        T: Send + 'static,
        F1: FnOnce(&Device, &Queue) -> T + Send + 'static,
        F2: FnOnce(&Device, &Queue) -> T + Send + 'static,
    {
        use tokio::task;

        let primary_device = self.primary_device.clone();
        let primary_queue = self.primary_queue.clone();

        let secondary_device = self.secondary_device.clone()
            .unwrap_or_else(|| self.primary_device.clone());
        let secondary_queue = self.secondary_queue.clone()
            .unwrap_or_else(|| self.primary_queue.clone());

        let primary_handle = task::spawn_blocking(move || {
            primary_task(&primary_device, &primary_queue)
        });

        let secondary_handle = task::spawn_blocking(move || {
            secondary_task(&secondary_device, &secondary_queue)
        });

        let (primary_result, secondary_result) = tokio::try_join!(primary_handle, secondary_handle)
            .map_err(|e| crate::GpuError::SyncError(e.to_string()))?;

        Ok((primary_result, secondary_result))
    }

    /// Chunked parallel processing for large datasets
    /// Splits data 77%/23% based on CU ratio (72:22)
    pub fn chunk_split_ratio(&self) -> (f32, f32) {
        if self.secondary_device.is_some() {
            (0.77, 0.23) // 72/(72+22), 22/(72+22)
        } else {
            (1.0, 0.0)
        }
    }
}

/// GPU assignment result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAssignment {
    /// Assign to primary GPU (RX 6800 XT)
    Primary,
    /// Assign to secondary GPU (RX 5500 XT)
    Secondary,
}

impl From<GpuPreference> for GpuAssignment {
    fn from(pref: GpuPreference) -> Self {
        match pref {
            GpuPreference::Primary | GpuPreference::Auto => GpuAssignment::Primary,
            GpuPreference::Secondary => GpuAssignment::Secondary,
        }
    }
}
