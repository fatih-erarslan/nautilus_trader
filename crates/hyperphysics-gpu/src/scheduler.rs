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

        // Optimized dispatch strategy:
        // 1. Try 1D dispatch if workgroups fit
        // 2. Use 2D dispatch for better cache locality on larger problems
        // 3. Fall back to 3D for extremely large dispatches
        self.optimize_grid_dimensions(n_workgroups)
    }

    /// Optimize grid dimensions for GPU cache locality and occupancy
    ///
    /// Uses tiled dispatch for better memory access patterns:
    /// - 1D for small workloads (cache-friendly sequential access)
    /// - 2D for medium workloads (improves L2 cache hit rate)
    /// - 3D for very large workloads (avoids dispatch limits)
    fn optimize_grid_dimensions(&self, n_workgroups: u32) -> [u32; 3] {
        // Threshold for switching to 2D (empirically 4096 gives good results)
        const THRESHOLD_2D: u32 = 4096;
        // Threshold for switching to 3D
        const THRESHOLD_3D: u64 = 16_777_216; // 4096 * 4096

        if n_workgroups <= self.max_dispatch_x && n_workgroups <= THRESHOLD_2D {
            // 1D dispatch: good for small, sequential workloads
            [n_workgroups, 1, 1]
        } else if (n_workgroups as u64) <= THRESHOLD_3D {
            // 2D dispatch: better cache locality for larger workloads
            // Choose dimensions that maximize L2 cache utilization
            let (x, y) = self.optimal_2d_factorization(n_workgroups);
            [x, y, 1]
        } else {
            // 3D dispatch: for extremely large workloads
            let (x, y, z) = self.optimal_3d_factorization(n_workgroups);
            [x, y, z]
        }
    }

    /// Find optimal 2D factorization for cache locality
    ///
    /// Prefers balanced dimensions (closer to square) for better
    /// spatial locality in memory access patterns.
    fn optimal_2d_factorization(&self, n: u32) -> (u32, u32) {
        // Start from sqrt and find nearest factor pair
        let sqrt_n = (n as f64).sqrt() as u32;

        // Search for factors near sqrt for balanced dimensions
        for delta in 0..=sqrt_n {
            let x = sqrt_n + delta;
            if x > self.max_dispatch_x {
                break;
            }

            // Check if x divides n evenly
            if n % x == 0 {
                let y = n / x;
                if y <= self.max_dispatch_y {
                    return (x, y);
                }
            }

            // Also check sqrt_n - delta
            if delta > 0 && sqrt_n >= delta {
                let x = sqrt_n - delta;
                if x > 0 && n % x == 0 {
                    let y = n / x;
                    if y <= self.max_dispatch_y {
                        return (x, y);
                    }
                }
            }
        }

        // Fallback: use ceiling division for approximate factorization
        let x = sqrt_n.max(1);
        let y = (n + x - 1) / x;

        // Ensure within limits
        let x = x.min(self.max_dispatch_x);
        let y = y.min(self.max_dispatch_y);

        (x, y)
    }

    /// Find optimal 3D factorization for very large dispatches
    fn optimal_3d_factorization(&self, n: u32) -> (u32, u32, u32) {
        // Find cube root for balanced 3D distribution
        let cbrt_n = (n as f64).powf(1.0 / 3.0).ceil() as u32;

        // Start with cubic distribution
        let x = cbrt_n.min(self.max_dispatch_x);
        let remaining = (n + x - 1) / x;

        // Factor remaining into y and z
        let (y, z) = self.optimal_2d_factorization(remaining);

        (x, y, z)
    }

    /// Compute dispatch for 2D grid (e.g., images, matrices)
    ///
    /// # Arguments
    /// * `width` - Grid width in elements
    /// * `height` - Grid height in elements
    /// * `tile_size` - Workgroup tile size (typically 8, 16, or 32)
    pub fn compute_dispatch_2d(&self, width: usize, height: usize, tile_size: u32) -> [u32; 3] {
        if width == 0 || height == 0 {
            return [0, 0, 0];
        }

        let x = ((width as u32) + tile_size - 1) / tile_size;
        let y = ((height as u32) + tile_size - 1) / tile_size;

        [x.min(self.max_dispatch_x), y.min(self.max_dispatch_y), 1]
    }

    /// Compute dispatch for 3D grid (e.g., volumes, 3D textures)
    ///
    /// # Arguments
    /// * `width` - Grid width in elements
    /// * `height` - Grid height in elements
    /// * `depth` - Grid depth in elements
    /// * `tile_size` - Workgroup tile size per dimension
    pub fn compute_dispatch_3d(
        &self,
        width: usize,
        height: usize,
        depth: usize,
        tile_size: u32,
    ) -> [u32; 3] {
        if width == 0 || height == 0 || depth == 0 {
            return [0, 0, 0];
        }

        let x = ((width as u32) + tile_size - 1) / tile_size;
        let y = ((height as u32) + tile_size - 1) / tile_size;
        let z = ((depth as u32) + tile_size - 1) / tile_size;

        [
            x.min(self.max_dispatch_x),
            y.min(self.max_dispatch_y),
            z.min(self.max_dispatch_z),
        ]
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

    #[test]
    fn test_2d_dispatch() {
        let scheduler = GPUScheduler::new(256);

        // 1024x1024 image with 16x16 tiles
        let dispatch = scheduler.compute_dispatch_2d(1024, 1024, 16);
        assert_eq!(dispatch, [64, 64, 1]);

        // Non-divisible dimensions
        let dispatch = scheduler.compute_dispatch_2d(1000, 800, 16);
        assert_eq!(dispatch, [63, 50, 1]); // ceil(1000/16) = 63, ceil(800/16) = 50

        // Edge cases
        assert_eq!(scheduler.compute_dispatch_2d(0, 100, 16), [0, 0, 0]);
        assert_eq!(scheduler.compute_dispatch_2d(100, 0, 16), [0, 0, 0]);
    }

    #[test]
    fn test_3d_dispatch() {
        let scheduler = GPUScheduler::new(256);

        // 256x256x256 volume with 8x8x8 tiles
        let dispatch = scheduler.compute_dispatch_3d(256, 256, 256, 8);
        assert_eq!(dispatch, [32, 32, 32]);

        // Non-cubic volume
        let dispatch = scheduler.compute_dispatch_3d(512, 256, 128, 8);
        assert_eq!(dispatch, [64, 32, 16]);

        // Edge cases
        assert_eq!(scheduler.compute_dispatch_3d(0, 100, 100, 8), [0, 0, 0]);
    }

    #[test]
    fn test_optimal_2d_factorization() {
        let scheduler = GPUScheduler::new(256);

        // Perfect square
        let (x, y) = scheduler.optimal_2d_factorization(10000);
        assert_eq!(x * y, 10000);
        assert!((x as i32 - y as i32).abs() <= (x as i32)); // Reasonably balanced

        // Prime number (no perfect factorization)
        let (x, y) = scheduler.optimal_2d_factorization(10007);
        assert!(x * y >= 10007); // Must cover all workgroups

        // Power of 2
        let (x, y) = scheduler.optimal_2d_factorization(4096);
        assert_eq!(x * y, 4096);
        assert_eq!(x, 64); // sqrt(4096) = 64
        assert_eq!(y, 64);
    }

    #[test]
    fn test_large_dispatch_uses_2d() {
        let scheduler = GPUScheduler::new(256);

        // Large dispatch should use 2D for better cache locality
        // 10M elements with 256-thread workgroups = ~40000 workgroups
        let dispatch = scheduler.compute_dispatch(10_000_000);

        // Should be 2D (y > 1) for large dispatches
        assert!(dispatch[0] > 1);
        assert!(dispatch[1] > 1 || dispatch[0] < 4096); // Either 2D or small enough for 1D
    }

    #[test]
    fn test_grid_dimensions_within_limits() {
        let scheduler = GPUScheduler::new(256);

        // Test various sizes to ensure we stay within limits
        for size in [1000, 100_000, 1_000_000, 10_000_000, 100_000_000_usize] {
            let dispatch = scheduler.compute_dispatch(size);

            assert!(dispatch[0] <= 65535, "x exceeds limit for size {}", size);
            assert!(dispatch[1] <= 65535, "y exceeds limit for size {}", size);
            assert!(dispatch[2] <= 65535, "z exceeds limit for size {}", size);

            // Total workgroups should cover all elements
            let total_workgroups = dispatch[0] as u64 * dispatch[1] as u64 * dispatch[2] as u64;
            let needed = ((size as u32 + 255) / 256) as u64;
            assert!(
                total_workgroups >= needed,
                "Not enough workgroups for size {}: {} < {}",
                size,
                total_workgroups,
                needed
            );
        }
    }
}
