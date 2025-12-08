// Matrix Multiplication Compute Shader for ruv_FANN GPU Acceleration
// Optimized for high-frequency trading neural network operations

struct Dimensions {
    rows_a: u32,
    cols_a: u32,
    rows_b: u32,
    cols_b: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    // Bounds checking
    if (row >= dims.rows_a || col >= dims.cols_b) {
        return;
    }
    
    var sum = 0.0;
    
    // Perform dot product for this cell
    for (var k = 0u; k < dims.cols_a; k++) {
        let a_idx = row * dims.cols_a + k;
        let b_idx = k * dims.cols_b + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    
    let c_idx = row * dims.cols_b + col;
    matrix_c[c_idx] = sum;
}

// Optimized version with shared memory for larger matrices
@compute @workgroup_size(16, 16)
fn main_optimized(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let tile_size = 16u;
    var shared_a: array<array<f32, 16>, 16>;
    var shared_b: array<array<f32, 16>, 16>;
    
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;
    
    var sum = 0.0;
    
    let num_tiles = (dims.cols_a + tile_size - 1u) / tile_size;
    
    for (var tile = 0u; tile < num_tiles; tile++) {
        // Load tile into shared memory
        let a_global_col = tile * tile_size + local_col;
        let b_global_row = tile * tile_size + local_row;
        
        if (row < dims.rows_a && a_global_col < dims.cols_a) {
            shared_a[local_row][local_col] = matrix_a[row * dims.cols_a + a_global_col];
        } else {
            shared_a[local_row][local_col] = 0.0;
        }
        
        if (b_global_row < dims.rows_b && col < dims.cols_b) {
            shared_b[local_row][local_col] = matrix_b[b_global_row * dims.cols_b + col];
        } else {
            shared_b[local_row][local_col] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial sum
        for (var k = 0u; k < tile_size; k++) {
            sum += shared_a[local_row][k] * shared_b[k][local_col];
        }
        
        workgroupBarrier();
    }
    
    if (row < dims.rows_a && col < dims.cols_b) {
        let c_idx = row * dims.cols_b + col;
        matrix_c[c_idx] = sum;
    }
}