// ReLU activation function shader
// Optimized for financial neural networks

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ActivationParams {
    size: u32,
    batch_size: u32,
    features: u32,
    alpha: f32,  // For Leaky ReLU
}

@group(1) @binding(0) var<uniform> params: ActivationParams;

@compute @workgroup_size(256, 1, 1)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    output[idx] = max(0.0, input[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn leaky_relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let x = input[idx];
    output[idx] = select(params.alpha * x, x, x > 0.0);
}

// ELU activation for better gradient flow
@compute @workgroup_size(256, 1, 1)
fn elu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let x = input[idx];
    output[idx] = select(params.alpha * (exp(x) - 1.0), x, x > 0.0);
}

// Swish activation (x * sigmoid(x))
@compute @workgroup_size(256, 1, 1)
fn swish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let x = input[idx];
    output[idx] = x / (1.0 + exp(-x));
}

// GELU activation for transformer models
@compute @workgroup_size(256, 1, 1)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let x = input[idx];
    let sqrt_2_pi = 0.7978845608;
    let gelu_approx = 0.5 * x * (1.0 + tanh(sqrt_2_pi * (x + 0.044715 * x * x * x)));
    output[idx] = gelu_approx;
}

// Vectorized ReLU for better performance
@compute @workgroup_size(64, 1, 1)
fn relu_vectorized(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x * 4u;
    
    if (idx + 3u >= params.size) {
        // Handle remaining elements individually
        for (var i: u32 = idx; i < min(idx + 4u, params.size); i++) {
            output[i] = max(0.0, input[i]);
        }
        return;
    }
    
    // Process 4 elements at once
    let input_vec = vec4<f32>(
        input[idx],
        input[idx + 1u],
        input[idx + 2u],
        input[idx + 3u]
    );
    
    let output_vec = max(vec4<f32>(0.0), input_vec);
    
    output[idx] = output_vec.x;
    output[idx + 1u] = output_vec.y;
    output[idx + 2u] = output_vec.z;
    output[idx + 3u] = output_vec.w;
}