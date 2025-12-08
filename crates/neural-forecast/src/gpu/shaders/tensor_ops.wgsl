// High-Performance Tensor Operations for GPU Acceleration
// Optimized for maximum throughput and minimal latency

struct TensorParams {
    rows: u32,
    cols: u32,
    depth: u32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> tensor_a: array<f32>;
@group(0) @binding(1) var<storage, read> tensor_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> tensor_c: array<f32>;
@group(0) @binding(3) var<uniform> params: TensorParams;

// Shared memory for tile-based matrix multiplication
var<workgroup> tile_a: array<f32, 1024>; // 32x32 tile
var<workgroup> tile_b: array<f32, 1024>; // 32x32 tile

// Vectorized matrix multiplication with tiling
@compute @workgroup_size(32, 32, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= params.rows || col >= params.cols) {
        return;
    }
    
    var sum = 0.0;
    let tile_size = 32u;
    let num_tiles = (params.depth + tile_size - 1) / tile_size;
    
    // Process tiles
    for (var tile_idx: u32 = 0; tile_idx < num_tiles; tile_idx++) {
        // Load tile A cooperatively
        let a_row = row;
        let a_col = tile_idx * tile_size + local_id.x;
        let a_idx = a_row * params.depth + a_col;
        
        if (a_col < params.depth) {
            tile_a[local_id.y * tile_size + local_id.x] = tensor_a[a_idx];
        } else {
            tile_a[local_id.y * tile_size + local_id.x] = 0.0;
        }
        
        // Load tile B cooperatively
        let b_row = tile_idx * tile_size + local_id.y;
        let b_col = col;
        let b_idx = b_row * params.cols + b_col;
        
        if (b_row < params.depth) {
            tile_b[local_id.y * tile_size + local_id.x] = tensor_b[b_idx];
        } else {
            tile_b[local_id.y * tile_size + local_id.x] = 0.0;
        }
        
        workgroupBarrier();
        
        // Compute partial dot product
        for (var k: u32 = 0; k < tile_size; k++) {
            sum += tile_a[local_id.y * tile_size + k] * tile_b[k * tile_size + local_id.x];
        }
        
        workgroupBarrier();
    }
    
    // Write result with alpha/beta scaling
    let c_idx = row * params.cols + col;
    tensor_c[c_idx] = params.alpha * sum + params.beta * tensor_c[c_idx];
}

// Fused elementwise operations for activation functions
@compute @workgroup_size(256, 1, 1)
fn fused_activations(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let idx = global_id.x;
    let total_elements = params.rows * params.cols;
    
    if (idx >= total_elements) {
        return;
    }
    
    let input_val = tensor_a[idx];
    var output_val: f32;
    
    // Switch based on params.alpha to select activation function
    if (params.alpha == 1.0) {
        // ReLU
        output_val = max(0.0, input_val);
    } else if (params.alpha == 2.0) {
        // Sigmoid
        output_val = 1.0 / (1.0 + exp(-input_val));
    } else if (params.alpha == 3.0) {
        // Tanh
        output_val = tanh(input_val);
    } else if (params.alpha == 4.0) {
        // GELU approximation
        output_val = 0.5 * input_val * (1.0 + tanh(sqrt(2.0 / 3.14159) * (input_val + 0.044715 * input_val * input_val * input_val)));
    } else if (params.alpha == 5.0) {
        // Swish/SiLU
        output_val = input_val / (1.0 + exp(-input_val));
    } else {
        // Linear (identity)
        output_val = input_val;
    }
    
    tensor_c[idx] = output_val;
}

// Optimized batch normalization
@compute @workgroup_size(256, 1, 1)
fn batch_norm(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let batch_size = params.rows;
    let feature_size = params.cols;
    
    if (idx >= batch_size * feature_size) {
        return;
    }
    
    let feature_idx = idx % feature_size;
    let batch_idx = idx / feature_size;
    
    // Compute mean and variance for this feature across batch
    var mean = 0.0;
    var variance = 0.0;
    
    // First pass: compute mean
    for (var b: u32 = 0; b < batch_size; b++) {
        let val_idx = b * feature_size + feature_idx;
        mean += tensor_a[val_idx];
    }
    mean /= f32(batch_size);
    
    // Second pass: compute variance
    for (var b: u32 = 0; b < batch_size; b++) {
        let val_idx = b * feature_size + feature_idx;
        let diff = tensor_a[val_idx] - mean;
        variance += diff * diff;
    }
    variance /= f32(batch_size);
    
    // Normalize
    let epsilon = 1e-5;
    let std_dev = sqrt(variance + epsilon);
    let normalized = (tensor_a[idx] - mean) / std_dev;
    
    // Scale and shift (gamma and beta are stored in tensor_b)
    let gamma = tensor_b[feature_idx];
    let beta = tensor_b[feature_idx + feature_size];
    
    tensor_c[idx] = gamma * normalized + beta;
}

// Efficient attention mechanism computation
@compute @workgroup_size(32, 32, 1)
fn attention_scores(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let seq_pos = global_id.x;
    let head_pos = global_id.y;
    let batch_idx = global_id.z;
    
    let seq_len = params.rows;
    let num_heads = params.cols;
    let head_dim = params.depth;
    
    if (seq_pos >= seq_len || head_pos >= num_heads) {
        return;
    }
    
    // Compute attention scores using efficient dot product
    var attention_score = 0.0;
    
    for (var d: u32 = 0; d < head_dim; d++) {
        let q_idx = batch_idx * seq_len * num_heads * head_dim + 
                   seq_pos * num_heads * head_dim + 
                   head_pos * head_dim + d;
        
        let k_idx = batch_idx * seq_len * num_heads * head_dim + 
                   seq_pos * num_heads * head_dim + 
                   head_pos * head_dim + d;
        
        attention_score += tensor_a[q_idx] * tensor_b[k_idx];
    }
    
    // Scale by sqrt(head_dim)
    attention_score /= sqrt(f32(head_dim));
    
    // Store attention score
    let out_idx = batch_idx * seq_len * num_heads + seq_pos * num_heads + head_pos;
    tensor_c[out_idx] = attention_score;
}

// Softmax computation with numerical stability
@compute @workgroup_size(256, 1, 1)
fn softmax_stable(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let batch_idx = global_id.x;
    let seq_len = params.rows;
    
    if (batch_idx >= params.cols) {
        return;
    }
    
    // Find maximum for numerical stability
    var max_val = -1e30;
    for (var i: u32 = 0; i < seq_len; i++) {
        let idx = batch_idx * seq_len + i;
        max_val = max(max_val, tensor_a[idx]);
    }
    
    // Compute exp(x - max) and sum
    var sum = 0.0;
    for (var i: u32 = 0; i < seq_len; i++) {
        let idx = batch_idx * seq_len + i;
        let exp_val = exp(tensor_a[idx] - max_val);
        tensor_c[idx] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (var i: u32 = 0; i < seq_len; i++) {
        let idx = batch_idx * seq_len + i;
        tensor_c[idx] /= sum;
    }
}

// Optimized layer normalization
@compute @workgroup_size(256, 1, 1)
fn layer_norm(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let sample_idx = global_id.x;
    let feature_size = params.cols;
    
    if (sample_idx >= params.rows) {
        return;
    }
    
    let start_idx = sample_idx * feature_size;
    
    // Compute mean
    var mean = 0.0;
    for (var i: u32 = 0; i < feature_size; i++) {
        mean += tensor_a[start_idx + i];
    }
    mean /= f32(feature_size);
    
    // Compute variance
    var variance = 0.0;
    for (var i: u32 = 0; i < feature_size; i++) {
        let diff = tensor_a[start_idx + i] - mean;
        variance += diff * diff;
    }
    variance /= f32(feature_size);
    
    // Normalize
    let epsilon = 1e-5;
    let std_dev = sqrt(variance + epsilon);
    
    for (var i: u32 = 0; i < feature_size; i++) {
        let idx = start_idx + i;
        let normalized = (tensor_a[idx] - mean) / std_dev;
        
        // Apply learnable parameters (gamma and beta from tensor_b)
        let gamma = tensor_b[i];
        let beta = tensor_b[i + feature_size];
        
        tensor_c[idx] = gamma * normalized + beta;
    }
}