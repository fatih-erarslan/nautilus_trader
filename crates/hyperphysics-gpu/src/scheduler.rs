//! GPU compute scheduler for work distribution
//!
//! Optimizes workgroup dispatch and batch sizes for GPU efficiency.

/// GPU work scheduler
pub struct GPUScheduler {
    workgroup_size: u32,
    max_dispatch_x: u32,
    max_dispatch_y: u32,
    max_dispatch_z: u32,
}

impl GPUScheduler {
    /// Create new scheduler with specified workgroup size
    ///
    /// # Arguments
    /// * `workgroup_size` - Threads per workgroup (typically 64, 128, or 256)
    pub fn new(workgroup_size: u32) -> Self {
        Self {
            workgroup_size,
            // WGPU default limits
            max_dispatch_x: 65535,
            max_dispatch_y: 65535,
            max_dispatch_z: 65535,
        }
    }

    /// Compute optimal workgroup dispatch for given element count
    ///
    /// Returns [x, y, z] workgroup counts
    ///
    /// # Example
    /// ```
    /// use hyperphysics_gpu::scheduler::GPUScheduler;
    /// let scheduler = GPUScheduler::new(256);
    /// let dispatch = scheduler.compute_dispatch(1000);
    /// // dispatch = [4, 1, 1] for 1000 elements with 256-thread workgroups
    /// ```
    pub fn compute_dispatch(&self, n_elements: usize) -> [u32; 3] {
        if n_elements == 0 {
            return [0, 0, 0];
        }

        // Compute number of workgroups needed
        let n_workgroups = (n_elements as u32 + self.workgroup_size - 1) / self.workgroup_size;

        // Simple 1D dispatch for now
        // TODO: Optimize for 2D/3D grids
        if n_workgroups <= self.max_dispatch_x {
            [n_workgroups, 1, 1]
        } else {
            // Split into 2D grid if too large
            let sqrt_n = (n_workgroups as f32).sqrt().ceil() as u32;
            [sqrt_n, sqrt_n, 1]
        }
    }

    /// Optimize batch size for given memory constraints
    ///
    /// # Arguments
    /// * `available_memory` - Available GPU memory in bytes
    /// * `element_size` - Size of each element in bytes
    ///
    /// # Returns
    /// Optimal batch size that fits in memory
    pub fn optimize_batch_size(&self, available_memory: u64, element_size: usize) -> usize {
        // Reserve 20% for overhead (buffers, parameters, etc.)
        let usable_memory = (available_memory as f64 * 0.8) as u64;

        // Calculate max elements that fit
        let max_elements = (usable_memory / element_size as u64) as usize;

        // Round down to multiple of workgroup size for efficiency
        (max_elements / self.workgroup_size as usize) * self.workgroup_size as usize
    }

    /// Estimate memory usage for given configuration
    ///
    /// # Arguments
    /// * `n_elements` - Number of elements to process
    /// * `element_size` - Size of each element in bytes
    /// * `n_buffers` - Number of buffers (e.g., 2 for double buffering)
    ///
    /// # Returns
    /// Estimated GPU memory usage in bytes
    pub fn estimate_memory_usage(
        &self,
        n_elements: usize,
        element_size: usize,
        n_buffers: usize,
    ) -> u64 {
        let data_size = (n_elements * element_size * n_buffers) as u64;

        // Add overhead estimates:
        // - Reduction buffers: ~1KB per workgroup
        // - Parameter buffers: ~1KB total
        // - Shader constants: ~1KB
        let n_workgroups = self.compute_dispatch(n_elements)[0] as u64;
        let overhead = n_workgroups * 1024 + 2048;

        data_size + overhead
    }

    /// Check if configuration fits in available memory
    pub fn fits_in_memory(
        &self,
        n_elements: usize,
        element_size: usize,
        n_buffers: usize,
        available_memory: u64,
    ) -> bool {
        self.estimate_memory_usage(n_elements, element_size, n_buffers) <= available_memory
    }

    /// Compute optimal work distribution for parallel reduction
    ///
    /// Returns (n_workgroups, reduction_factor) where:
    /// - n_workgroups: Number of workgroups for first pass
    /// - reduction_factor: Elements per thread in reduction
    pub fn compute_reduction_strategy(&self, n_elements: usize) -> (usize, usize) {
        let n_workgroups = self.compute_dispatch(n_elements)[0] as usize;
        let elements_per_workgroup = (n_elements + n_workgroups - 1) / n_workgroups;

        (n_workgroups, elements_per_workgroup)
    }
}

impl Default for GPUScheduler {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_calculation() {
        let scheduler = GPUScheduler::new(256);

        // Small dispatch
        assert_eq!(scheduler.compute_dispatch(256), [1, 1, 1]);
        assert_eq!(scheduler.compute_dispatch(512), [2, 1, 1]);
        assert_eq!(scheduler.compute_dispatch(1000), [4, 1, 1]);

        // Edge cases
        assert_eq!(scheduler.compute_dispatch(0), [0, 0, 0]);
        assert_eq!(scheduler.compute_dispatch(1), [1, 1, 1]);
        assert_eq!(scheduler.compute_dispatch(255), [1, 1, 1]);
        assert_eq!(scheduler.compute_dispatch(257), [2, 1, 1]);
    }

    #[test]
    fn test_batch_optimization() {
        let scheduler = GPUScheduler::new(256);

        // 1GB available, 64-byte elements
        let batch_size = scheduler.optimize_batch_size(1_000_000_000, 64);

        // Should be large but aligned to workgroup size
        assert!(batch_size > 0);
        assert_eq!(batch_size % 256, 0);

        // Should fit in 80% of available memory
        let memory_used = (batch_size * 64) as u64;
        assert!(memory_used <= 800_000_000);
    }

    #[test]
    fn test_memory_estimation() {
        let scheduler = GPUScheduler::new(256);

        // 1M elements × 64 bytes × 2 buffers = 128MB + overhead
        let estimated = scheduler.estimate_memory_usage(1_000_000, 64, 2);

        let data_size = 1_000_000 * 64 * 2;
        assert!(estimated >= data_size as u64);
        assert!(estimated < (data_size as u64 * 2)); // Overhead shouldn't double size
    }

    #[test]
    fn test_memory_fit_check() {
        let scheduler = GPUScheduler::new(256);

        // Should fit: 1K elements × 64 bytes × 2 buffers = 128KB
        assert!(scheduler.fits_in_memory(1000, 64, 2, 1_000_000));

        // Should not fit: 1M elements × 64 bytes × 2 buffers = 128MB
        assert!(!scheduler.fits_in_memory(1_000_000, 64, 2, 1_000_000));
    }

    #[test]
    fn test_reduction_strategy() {
        let scheduler = GPUScheduler::new(256);

        let (n_workgroups, elements_per_wg) = scheduler.compute_reduction_strategy(100_000);

        // Should use multiple workgroups
        assert!(n_workgroups > 1);

        // Each workgroup should handle reasonable number of elements
        assert!(elements_per_wg > 0);
        assert!(elements_per_wg <= 1000); // Not too many per workgroup
    }
}
