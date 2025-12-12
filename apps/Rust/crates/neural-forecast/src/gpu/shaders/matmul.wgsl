// Matrix multiplication shader for neural networks
// Optimized for financial time series data processing

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

// Uniforms for matrix dimensions
struct MatMulParams {
    m: u32,      // rows of A, rows of result
    k: u32,      // cols of A, rows of B
    n: u32,      // cols of B, cols of result
    batch_size: u32,
}

@group(1) @binding(0) var<uniform> params: MatMulParams;

// Local memory for tile-based multiplication
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;
    
    // Early exit if out of bounds
    if (row >= params.m || col >= params.n || batch >= params.batch_size) {
        return;
    }
    
    let local_row = local_id.x;
    let local_col = local_id.y;
    
    var sum: f32 = 0.0;
    
    // Tile-based matrix multiplication for better cache utilization
    let num_tiles = (params.k + 15u) / 16u;
    
    for (var tile_idx: u32 = 0u; tile_idx < num_tiles; tile_idx++) {
        // Load tile of matrix A
        let a_col = tile_idx * 16u + local_col;
        if (a_col < params.k) {
            let a_idx = batch * params.m * params.k + row * params.k + a_col;
            tile_a[local_row][local_col] = matrix_a[a_idx];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }
        
        // Load tile of matrix B
        let b_row = tile_idx * 16u + local_row;
        if (b_row < params.k) {
            let b_idx = batch * params.k * params.n + b_row * params.n + col;
            tile_b[local_row][local_col] = matrix_b[b_idx];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }
        
        // Synchronize workgroup
        workgroupBarrier();
        
        // Compute partial sum for this tile
        for (var i: u32 = 0u; i < 16u; i++) {
            sum += tile_a[local_row][i] * tile_b[i][local_col];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Write result
    let result_idx = batch * params.m * params.n + row * params.n + col;
    result[result_idx] = sum;
}

// Optimized version for specific matrix sizes common in neural networks
@compute @workgroup_size(32, 8, 1)
fn matmul_optimized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;
    
    if (row >= params.m || col >= params.n || batch >= params.batch_size) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Unrolled loop for common neural network dimensions
    for (var k: u32 = 0u; k < params.k; k++) {
        let a_idx = batch * params.m * params.k + row * params.k + k;
        let b_idx = batch * params.k * params.n + k * params.n + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    
    let result_idx = batch * params.m * params.n + row * params.n + col;
    result[result_idx] = sum;
}

// Batch matrix multiplication for multiple assets
@compute @workgroup_size(16, 16, 1)
fn batch_matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let asset_id = global_id.z;
    
    if (row >= params.m || col >= params.n || asset_id >= params.batch_size) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Vectorized computation for better performance
    let k_vec4 = params.k / 4u;
    let k_remainder = params.k % 4u;
    
    // Process 4 elements at a time
    for (var k_idx: u32 = 0u; k_idx < k_vec4; k_idx++) {
        let k_base = k_idx * 4u;
        
        // Load 4 elements from matrix A
        let a_idx_base = asset_id * params.m * params.k + row * params.k + k_base;
        let a_vec = vec4<f32>(
            matrix_a[a_idx_base],
            matrix_a[a_idx_base + 1u],
            matrix_a[a_idx_base + 2u],
            matrix_a[a_idx_base + 3u]
        );
        
        // Load 4 elements from matrix B
        let b_idx_base = asset_id * params.k * params.n + k_base * params.n + col;
        let b_vec = vec4<f32>(
            matrix_b[b_idx_base],
            matrix_b[b_idx_base + params.n],
            matrix_b[b_idx_base + 2u * params.n],
            matrix_b[b_idx_base + 3u * params.n]
        );
        
        // Dot product
        sum += dot(a_vec, b_vec);
    }
    
    // Handle remaining elements
    for (var k: u32 = k_vec4 * 4u; k < params.k; k++) {
        let a_idx = asset_id * params.m * params.k + row * params.k + k;
        let b_idx = asset_id * params.k * params.n + k * params.n + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    
    let result_idx = asset_id * params.m * params.n + row * params.n + col;
    result[result_idx] = sum;
}