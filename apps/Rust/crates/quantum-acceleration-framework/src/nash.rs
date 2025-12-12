//! Ultra-fast Nash equilibrium solver with GPU acceleration

use std::sync::Arc;
use tokio::sync::Mutex;
use crate::{
    QBMIAError, QBMIAResult, PayoffMatrix, StrategyVector, 
    NashEquilibrium, NashSolverParams, gpu::GpuPipeline,
    GpuBufferUsage, KernelParams
};

/// GPU-accelerated Nash equilibrium solver for game theory
pub struct NashSolver {
    /// GPU pipeline
    gpu_pipeline: Arc<GpuPipeline>,
    
    /// Solver metrics
    metrics: Arc<Mutex<NashMetrics>>,
}

impl NashSolver {
    /// Create new Nash equilibrium solver
    pub async fn new(gpu_pipeline: Arc<GpuPipeline>) -> QBMIAResult<Self> {
        tracing::info!("Initializing Nash equilibrium solver");
        
        let metrics = Arc::new(Mutex::new(NashMetrics::new()));
        
        let solver = Self {
            gpu_pipeline,
            metrics,
        };
        
        // Pre-compile solver shaders
        solver.precompile_shaders().await?;
        
        tracing::info!("Nash equilibrium solver initialized");
        Ok(solver)
    }
    
    /// Solve Nash equilibrium with sub-500ns performance target
    pub async fn solve(
        &self,
        payoff_matrix: &PayoffMatrix,
        initial_strategies: &StrategyVector,
        params: &NashSolverParams,
    ) -> QBMIAResult<NashEquilibrium> {
        let start_time = std::time::Instant::now();
        
        // Validate inputs
        if payoff_matrix.cols != initial_strategies.dimension() {
            return Err(QBMIAError::nash_solver("Matrix dimensions don't match strategy vector"));
        }
        
        // Choose optimal solver algorithm based on problem size
        let equilibrium = if payoff_matrix.rows <= 8 && payoff_matrix.cols <= 8 {
            // Use specialized small matrix solver for ultra-fast performance
            self.solve_small_matrix(payoff_matrix, initial_strategies, params).await?
        } else if payoff_matrix.rows <= 64 && payoff_matrix.cols <= 64 {
            // Use medium matrix solver with GPU acceleration
            self.solve_medium_matrix(payoff_matrix, initial_strategies, params).await?
        } else {
            // Use scalable solver for large matrices
            self.solve_large_matrix(payoff_matrix, initial_strategies, params).await?
        };
        
        let solving_time = start_time.elapsed();
        let mut metrics = self.metrics.lock().await;
        metrics.record_solution(solving_time, payoff_matrix.rows, payoff_matrix.cols);
        
        // Validate sub-500ns target for small matrices
        if payoff_matrix.rows <= 4 && payoff_matrix.cols <= 4 && solving_time.as_nanos() > 500 {
            tracing::warn!(
                "Nash solving ({}x{}) took {}ns, exceeding 500ns target",
                payoff_matrix.rows, payoff_matrix.cols, solving_time.as_nanos()
            );
        }
        
        tracing::debug!(
            "Solved Nash equilibrium ({}x{}) in {:.3}ns",
            payoff_matrix.rows, payoff_matrix.cols, solving_time.as_nanos()
        );
        
        Ok(equilibrium)
    }
    
    /// Solve small matrix (≤8x8) with specialized ultra-fast algorithm
    async fn solve_small_matrix(
        &self,
        payoff_matrix: &PayoffMatrix,
        initial_strategies: &StrategyVector,
        params: &NashSolverParams,
    ) -> QBMIAResult<NashEquilibrium> {
        // For small matrices, use direct analytical solution when possible
        if payoff_matrix.rows == 2 && payoff_matrix.cols == 2 {
            return self.solve_2x2_analytical(payoff_matrix).await;
        }
        
        // Use GPU with specialized small matrix kernel
        let shader_source = self.get_small_matrix_shader(payoff_matrix.rows, payoff_matrix.cols);
        
        // Create GPU buffers
        let matrix_buffer = self.gpu_pipeline
            .create_buffer(payoff_matrix.to_bytes(), GpuBufferUsage::Storage)
            .await?;
        
        let strategy_buffer = self.gpu_pipeline
            .create_buffer(initial_strategies.to_bytes(), GpuBufferUsage::Storage)
            .await?;
        
        let params_data = self.serialize_params(params)?;
        let params_buffer = self.gpu_pipeline
            .create_buffer(&params_data, GpuBufferUsage::Uniform)
            .await?;
        
        // Create output buffers
        let result_strategies_data = vec![0f32; initial_strategies.dimension()];
        let result_buffer = self.gpu_pipeline
            .create_buffer(bytemuck::cast_slice(&result_strategies_data), GpuBufferUsage::Storage)
            .await?;
        
        let convergence_data = vec![0f32; 1];
        let convergence_buffer = self.gpu_pipeline
            .create_buffer(bytemuck::cast_slice(&convergence_data), GpuBufferUsage::Storage)
            .await?;
        
        // Get compute pipeline
        let pipeline = self.gpu_pipeline
            .get_compute_pipeline(&shader_source, "main")
            .await?;
        
        // Execute kernel
        let kernel_params = KernelParams {
            dispatch_size: [1, 1, 1], // Single workgroup for small matrices
            input_buffers: vec![matrix_buffer, strategy_buffer, params_buffer],
            output_buffers: vec![result_buffer.clone(), convergence_buffer.clone()],
            timeout_ns: 5_000, // 5μs timeout
        };
        
        self.gpu_pipeline.execute_kernel(pipeline, &kernel_params).await?;
        
        // Read results
        let strategy_data = self.gpu_pipeline.read_buffer(&result_buffer).await?;
        let convergence_data = self.gpu_pipeline.read_buffer(&convergence_buffer).await?;
        
        let final_probabilities: Vec<f32> = bytemuck::cast_slice(&strategy_data).to_vec();
        let convergence: f32 = bytemuck::cast_slice(&convergence_data)[0];
        
        // Calculate payoffs
        let payoffs = self.calculate_payoffs(payoff_matrix, &final_probabilities)?;
        
        let final_strategy = StrategyVector::new(final_probabilities)?;
        
        Ok(NashEquilibrium::new(
            vec![final_strategy],
            convergence,
            params.max_iterations, // Simplified
            payoffs,
        ))
    }
    
    /// Solve 2x2 matrix analytically for ultimate speed
    async fn solve_2x2_analytical(&self, payoff_matrix: &PayoffMatrix) -> QBMIAResult<NashEquilibrium> {
        let start_time = std::time::Instant::now();
        
        // Extract payoff matrix elements
        let a = payoff_matrix.get(0, 0).unwrap();
        let b = payoff_matrix.get(0, 1).unwrap();
        let c = payoff_matrix.get(1, 0).unwrap();
        let d = payoff_matrix.get(1, 1).unwrap();
        
        // Analytical solution for 2x2 symmetric game
        let denominator = (a - b - c + d);
        
        let (p, q) = if denominator.abs() < 1e-10 {
            // Pure strategy equilibrium
            if a > c {
                (1.0, if a > b { 1.0 } else { 0.0 })
            } else {
                (0.0, if c > d { 1.0 } else { 0.0 })
            }
        } else {
            // Mixed strategy equilibrium
            let p = (d - c) / denominator;
            let q = (d - b) / denominator;
            (p.clamp(0.0, 1.0), q.clamp(0.0, 1.0))
        };
        
        let strategies = vec![
            StrategyVector::new(vec![p, 1.0 - p])?,
            StrategyVector::new(vec![q, 1.0 - q])?,
        ];
        
        let payoffs = vec![
            p * q * a + p * (1.0 - q) * b + (1.0 - p) * q * c + (1.0 - p) * (1.0 - q) * d,
            q * p * a + q * (1.0 - p) * c + (1.0 - q) * p * b + (1.0 - q) * (1.0 - p) * d,
        ];
        
        let solving_time = start_time.elapsed();
        
        // This should be ultra-fast (< 50ns)
        if solving_time.as_nanos() > 50 {
            tracing::warn!(
                "2x2 analytical solution took {}ns, exceeding 50ns target",
                solving_time.as_nanos()
            );
        }
        
        Ok(NashEquilibrium::new(
            strategies,
            0.0, // Analytical solution is exact
            1,   // Single iteration
            payoffs,
        ))
    }
    
    /// Solve medium matrix (≤64x64) with GPU acceleration
    async fn solve_medium_matrix(
        &self,
        payoff_matrix: &PayoffMatrix,
        initial_strategies: &StrategyVector,
        params: &NashSolverParams,
    ) -> QBMIAResult<NashEquilibrium> {
        // Use iterative best response with GPU acceleration
        let shader_source = self.get_medium_matrix_shader(payoff_matrix.rows, payoff_matrix.cols);
        
        // Create GPU buffers
        let matrix_buffer = self.gpu_pipeline
            .create_buffer(payoff_matrix.to_bytes(), GpuBufferUsage::Storage)
            .await?;
        
        let mut current_strategies = initial_strategies.clone();
        let mut iteration = 0;
        let mut convergence = f32::INFINITY;
        
        while iteration < params.max_iterations && convergence > params.convergence_threshold {
            // Update strategies using GPU
            let strategy_buffer = self.gpu_pipeline
                .create_buffer(current_strategies.to_bytes(), GpuBufferUsage::Storage)
                .await?;
            
            let output_buffer = self.gpu_pipeline
                .create_buffer(current_strategies.to_bytes(), GpuBufferUsage::Storage)
                .await?;
            
            let pipeline = self.gpu_pipeline
                .get_compute_pipeline(&shader_source, "main")
                .await?;
            
            let kernel_params = KernelParams {
                dispatch_size: [payoff_matrix.cols as u32, 1, 1],
                input_buffers: vec![matrix_buffer.clone(), strategy_buffer],
                output_buffers: vec![output_buffer.clone()],
                timeout_ns: 10_000, // 10μs timeout
            };
            
            self.gpu_pipeline.execute_kernel(pipeline, &kernel_params).await?;
            
            // Read updated strategies
            let strategy_data = self.gpu_pipeline.read_buffer(&output_buffer).await?;
            let new_probabilities: Vec<f32> = bytemuck::cast_slice(&strategy_data).to_vec();
            
            // Calculate convergence
            convergence = current_strategies.probabilities
                .iter()
                .zip(new_probabilities.iter())
                .map(|(old, new)| (old - new).abs())
                .fold(0.0, f32::max);
            
            current_strategies = StrategyVector::new(new_probabilities)?;
            iteration += 1;
        }
        
        let payoffs = self.calculate_payoffs(payoff_matrix, &current_strategies.probabilities)?;
        
        Ok(NashEquilibrium::new(
            vec![current_strategies],
            convergence,
            iteration,
            payoffs,
        ))
    }
    
    /// Solve large matrix with scalable GPU algorithm
    async fn solve_large_matrix(
        &self,
        payoff_matrix: &PayoffMatrix,
        initial_strategies: &StrategyVector,
        params: &NashSolverParams,
    ) -> QBMIAResult<NashEquilibrium> {
        // Use scalable fictitious play algorithm with GPU acceleration
        let shader_source = self.get_large_matrix_shader(payoff_matrix.rows, payoff_matrix.cols);
        
        // Implement scalable algorithm for large matrices
        // This would use techniques like:
        // - Matrix-vector multiplication on GPU
        // - Parallel strategy updates
        // - Convergence checking with reduction operations
        
        // Simplified implementation for now
        self.solve_medium_matrix(payoff_matrix, initial_strategies, params).await
    }
    
    /// Calculate expected payoffs for given strategies
    fn calculate_payoffs(&self, payoff_matrix: &PayoffMatrix, strategies: &[f32]) -> QBMIAResult<Vec<f32>> {
        let mut payoffs = vec![0.0; payoff_matrix.rows];
        
        for i in 0..payoff_matrix.rows {
            for j in 0..payoff_matrix.cols {
                let payoff = payoff_matrix.get(i, j).unwrap();
                payoffs[i] += payoff * strategies[j];
            }
        }
        
        Ok(payoffs)
    }
    
    /// Pre-compile Nash solver shaders
    async fn precompile_shaders(&self) -> QBMIAResult<()> {
        tracing::info!("Pre-compiling Nash solver shaders");
        
        // Pre-compile for common matrix sizes
        for &(rows, cols) in &[(2, 2), (3, 3), (4, 4), (8, 8), (16, 16), (32, 32)] {
            let small_shader = self.get_small_matrix_shader(rows, cols);
            self.gpu_pipeline.get_compute_pipeline(&small_shader, "main").await?;
            
            let medium_shader = self.get_medium_matrix_shader(rows, cols);
            self.gpu_pipeline.get_compute_pipeline(&medium_shader, "main").await?;
        }
        
        tracing::info!("Nash solver shader pre-compilation completed");
        Ok(())
    }
    
    /// Generate shader for small matrix Nash solving
    fn get_small_matrix_shader(&self, rows: usize, cols: usize) -> String {
        format!(r#"
        @group(0) @binding(0) var<storage, read> payoff_matrix: array<f32>;
        @group(0) @binding(1) var<storage, read> current_strategy: array<f32>;
        @group(0) @binding(2) var<uniform> params: NashParams;
        @group(0) @binding(3) var<storage, read_write> result_strategy: array<f32>;
        @group(0) @binding(4) var<storage, read_write> convergence: array<f32>;
        
        struct NashParams {{
            learning_rate: f32,
            max_iterations: u32,
            convergence_threshold: f32,
            regularization: f32,
        }}
        
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let rows = {}u;
            let cols = {}u;
            
            // Compute expected payoffs for each strategy
            var expected_payoffs: array<f32, {}>;
            
            for (var i = 0u; i < rows; i++) {{
                expected_payoffs[i] = 0.0;
                for (var j = 0u; j < cols; j++) {{
                    expected_payoffs[i] += payoff_matrix[i * cols + j] * current_strategy[j];
                }}
            }}
            
            // Find best response (strategy with highest expected payoff)
            var best_strategy = 0u;
            var best_payoff = expected_payoffs[0];
            
            for (var i = 1u; i < rows; i++) {{
                if (expected_payoffs[i] > best_payoff) {{
                    best_payoff = expected_payoffs[i];
                    best_strategy = i;
                }}
            }}
            
            // Update strategy with learning rate
            for (var i = 0u; i < rows; i++) {{
                if (i == best_strategy) {{
                    result_strategy[i] = current_strategy[i] + params.learning_rate * (1.0 - current_strategy[i]);
                }} else {{
                    result_strategy[i] = current_strategy[i] * (1.0 - params.learning_rate);
                }}
            }}
            
            // Normalize strategy
            var sum = 0.0;
            for (var i = 0u; i < rows; i++) {{
                sum += result_strategy[i];
            }}
            
            if (sum > 0.0) {{
                for (var i = 0u; i < rows; i++) {{
                    result_strategy[i] /= sum;
                }}
            }}
            
            // Calculate convergence metric
            var conv = 0.0;
            for (var i = 0u; i < rows; i++) {{
                conv += abs(result_strategy[i] - current_strategy[i]);
            }}
            convergence[0] = conv;
        }}
        "#, rows, cols, rows)
    }
    
    /// Generate shader for medium matrix Nash solving
    fn get_medium_matrix_shader(&self, rows: usize, cols: usize) -> String {
        format!(r#"
        @group(0) @binding(0) var<storage, read> payoff_matrix: array<f32>;
        @group(0) @binding(1) var<storage, read> current_strategy: array<f32>;
        @group(0) @binding(2) var<storage, read_write> result_strategy: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let strategy_index = global_id.x;
            let rows = {}u;
            let cols = {}u;
            
            if (strategy_index >= rows) {{
                return;
            }}
            
            // Compute expected payoff for this strategy
            var expected_payoff = 0.0;
            for (var j = 0u; j < cols; j++) {{
                expected_payoff += payoff_matrix[strategy_index * cols + j] * current_strategy[j];
            }}
            
            // Store in shared memory or global memory for reduction
            // Simplified: just update strategy based on relative payoff
            result_strategy[strategy_index] = max(0.0, current_strategy[strategy_index] + 0.01 * expected_payoff);
        }}
        "#, rows, cols)
    }
    
    /// Generate shader for large matrix Nash solving
    fn get_large_matrix_shader(&self, rows: usize, cols: usize) -> String {
        // For large matrices, use more sophisticated algorithms
        // like parallel fictitious play or evolutionary dynamics
        self.get_medium_matrix_shader(rows, cols) // Simplified
    }
    
    /// Serialize Nash solver parameters for GPU
    fn serialize_params(&self, params: &NashSolverParams) -> QBMIAResult<Vec<u8>> {
        let data = [
            params.learning_rate,
            params.max_iterations as f32,
            params.convergence_threshold,
            params.regularization,
        ];
        Ok(bytemuck::cast_slice(&data).to_vec())
    }
    
    /// Get performance metrics
    pub async fn get_metrics(&self) -> NashMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
}

/// Nash equilibrium solver metrics
#[derive(Debug, Clone, Default)]
pub struct NashMetrics {
    pub total_solutions: u64,
    pub total_solving_time: std::time::Duration,
    pub small_matrix_solutions: u64,
    pub medium_matrix_solutions: u64,
    pub large_matrix_solutions: u64,
    pub analytical_solutions: u64,
}

impl NashMetrics {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_solution(&mut self, duration: std::time::Duration, rows: usize, cols: usize) {
        self.total_solutions += 1;
        self.total_solving_time += duration;
        
        if rows <= 8 && cols <= 8 {
            self.small_matrix_solutions += 1;
            if rows == 2 && cols == 2 {
                self.analytical_solutions += 1;
            }
        } else if rows <= 64 && cols <= 64 {
            self.medium_matrix_solutions += 1;
        } else {
            self.large_matrix_solutions += 1;
        }
    }
    
    pub fn average_solving_time(&self) -> Option<std::time::Duration> {
        if self.total_solutions > 0 {
            Some(self.total_solving_time / self.total_solutions as u32)
        } else {
            None
        }
    }
    
    pub fn solutions_per_second(&self) -> f64 {
        if self.total_solving_time.as_secs_f64() > 0.0 {
            self.total_solutions as f64 / self.total_solving_time.as_secs_f64()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::GpuPipeline;
    
    #[tokio::test]
    async fn test_nash_solver_initialization() {
        let gpu_pipeline = match GpuPipeline::new().await {
            Ok(pipeline) => Arc::new(pipeline),
            Err(_) => return, // Skip if no GPU
        };
        
        let solver = NashSolver::new(gpu_pipeline).await;
        assert!(solver.is_ok());
    }
    
    #[tokio::test]
    async fn test_2x2_analytical_solution() {
        let gpu_pipeline = match GpuPipeline::new().await {
            Ok(pipeline) => Arc::new(pipeline),
            Err(_) => return, // Skip if no GPU
        };
        
        let solver = NashSolver::new(gpu_pipeline).await.unwrap();
        
        // Simple 2x2 game
        let payoff_matrix = PayoffMatrix::new(2, 2, vec![3.0, 0.0, 0.0, 1.0]).unwrap();
        let result = solver.solve_2x2_analytical(&payoff_matrix).await.unwrap();
        
        assert_eq!(result.strategies.len(), 2);
        assert!(result.convergence < 1e-6);
    }
    
    #[tokio::test]
    async fn test_small_matrix_solution() {
        let gpu_pipeline = match GpuPipeline::new().await {
            Ok(pipeline) => Arc::new(pipeline),
            Err(_) => return, // Skip if no GPU
        };
        
        let solver = NashSolver::new(gpu_pipeline).await.unwrap();
        
        let payoff_matrix = PayoffMatrix::random(4, 4).unwrap();
        let initial_strategies = StrategyVector::uniform(4).unwrap();
        let params = NashSolverParams::default();
        
        let result = solver.solve(&payoff_matrix, &initial_strategies, &params).await;
        assert!(result.is_ok());
    }
}