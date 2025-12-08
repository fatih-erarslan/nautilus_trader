// Tanh Activation Function Compute Shader for ruv_FANN GPU Acceleration
// Optimized for high-frequency trading neural network operations

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Optimized tanh computation using exp
    // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    // For numerical stability, we use different formulations based on input magnitude
    
    if (abs(x) > 10.0) {
        // For large values, tanh approaches ±1
        output_data[index] = sign(x);
    } else if (abs(x) < 0.001) {
        // For small values, use linear approximation: tanh(x) ≈ x
        output_data[index] = x;
    } else {
        // Standard computation
        let exp_2x = exp(2.0 * x);
        output_data[index] = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
}

// Alternative implementation using the identity: tanh(x) = 2*sigmoid(2x) - 1
@compute @workgroup_size(256)
fn tanh_sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    let sigmoid_2x = 1.0 / (1.0 + exp(-2.0 * x));
    output_data[index] = 2.0 * sigmoid_2x - 1.0;
}

// Fast approximation using rational function
@compute @workgroup_size(256)
fn tanh_fast(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    // Fast tanh approximation: x / (1 + |x|)
    // This is less accurate but much faster
    if (abs(x) > 5.0) {
        output_data[index] = sign(x);
    } else {
        output_data[index] = x / (1.0 + abs(x));
    }
}

// High precision tanh with improved numerical stability
@compute @workgroup_size(256)
fn tanh_precise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let x = input_data[index];
    
    if (abs(x) > 15.0) {
        // For very large values, tanh saturates
        output_data[index] = sign(x);
    } else {
        // Use symmetric property: tanh(-x) = -tanh(x)
        let abs_x = abs(x);
        
        if (abs_x < 0.0001) {
            // Linear approximation for very small values
            output_data[index] = x;
        } else {
            // Compute for positive value and apply sign
            let exp_2x = exp(2.0 * abs_x);
            let tanh_abs_x = (exp_2x - 1.0) / (exp_2x + 1.0);
            output_data[index] = sign(x) * tanh_abs_x;
        }
    }
}

// Derivative of tanh for backpropagation
@compute @workgroup_size(256)
fn tanh_derivative(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    if (index >= arrayLength(&input_data)) {
        return;
    }
    
    let tanh_x = input_data[index]; // Assuming input is already tanh(x)
    // d/dx tanh(x) = 1 - tanh²(x)
    output_data[index] = 1.0 - tanh_x * tanh_x;
}